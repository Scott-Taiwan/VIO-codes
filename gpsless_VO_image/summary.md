# gpsless_VO_image — 專案摘要

無 GPS 環境下，用下視相機拍到的連續照片估算 F450／Jetson Orin Nano 無人機的位移（Visual Odometry），並將估算結果接進 Pixhawk 作為第二組 GPS 來源。跟 `gpsless_mapping`／`gpsless_superpoint`（用衛星圖磚比對做「絕對定位」）是互補關係：這裡做的是兩次定位之間的「相對位移」估算。

## 核心演算法 — `visual_odometry.py`

流程（詳見 [flow.md](flow.md) 的完整圖解）：

1. **`list_images()` / `parse_altitude_m()`** — 依拍攝時間排序照片，並從檔名解析每張的高度。
2. **`detect_and_match()`** — 先做中央裁切（濾掉魚眼鏡頭邊緣畸變），再用 SIFT／ORB 偵測特徵並比對。
3. **`estimate_similarity()`** — RANSAC 擬合「旋轉＋縮放＋平移」的相似變換。
4. **`displacement_from_transform()`** — 用高度＋相機視角換算像素位移為公尺。
5. **信心度把關** — inlier 數量與 inlier 比例都太低時，標記為 `LOW-CONFIDENCE`／`FAIL`，不會假裝有把握。

**數學已驗證正確**：用已知的合成位移（20m／-10m／15°）反推，誤差在 3 公分內。

## 實測結果與已知限制

用 `photo_obtained/` 裡 3 張真實照片測試（檔名內建的 GPS 座標僅作驗證用，從未餵進演算法）：

| 路段 | VO 估算 | GPS 真實距離 | 誤差 | 信心 |
|---|---|---|---|---|
| B→A | 7.07 m | 47.24 m | 85% | LOW-CONFIDENCE |
| A→C | 3.96 m | 9.48 m | 58% | LOW-CONFIDENCE |

**根本原因**：這 3 張照片間隔太遠（60m 高度、視野只有約 70×55m），兩張照片重疊區域太小，加上草地與跑道圈狀標記等重複性紋理，讓 RANSAC 鎖定一個自洽但錯誤的變換。**結論：拍照間隔必須夠密（重疊度夠高）VO 才準**，稀疏的「每個航點拍一張」用不了。

## Pixhawk 整合

沿用既有的 `gpsless_mapping/gps_sim.py` `GPS_INPUT`（MAVLink 232，GPS2，`GPS2_TYPE=14`）機制，沒有另外開新通道。

- **`dead_reckoning.py`** — `PositionTracker`：持有一個「錨點」（絕對定位：lat/lon/alt + 真實航向），用每一段 VO 結果把它往前推。信心不足或失敗的路段會讓 `hacc_m`（水平精度）成長更多，讓 `GPS_AUTO_SWITCH` 自動偏向較新的絕對定位。旋轉數學已用合成案例驗證（航向 0° 與 90° 兩種情況）。
- **`gps_sim.py`**（微調）— `send_gps_input()` 新增 `hacc` 參數（預設 0.5，向下相容），是信心標記真正接進去的地方。
- **`vo_gps_bridge.py`** — 批次版驅動程式：讀一個資料夾的照片，逐段跑 VO → dead-reckoning → 送 `GPS_INPUT`，含 `--dry-run`。

## 即時拍照迴圈 — `live_vo_gps.py`

開一次相機、依固定 `--interval` 秒數持續拍照，每張新照片跟前一張跑 VO（用新增的 `visual_odometry.process_pair_arrays()`，記憶體版本、不用每張都寫檔），結果餵進 `PositionTracker`，兩次拍照之間持續以 `--gps-hz` 頻率重送目前位置給 Pixhawk（避免 GPS2 被當成逾時）。錨點（座標／高度／航向）預設自動從 Pixhawk 遙測（`GLOBAL_POSITION_INT`、`VFR_HUD`）讀取，讀不到才退回手動參數或 `gps_sim.py` 快取的 `last_home.json`。

**已在真實硬體上驗證**（這台開發機本身就是裝在 F450 上的 Jetson Orin Nano）：相機能正常開啟並拍到真實影像，`--dry-run` 模式下整條「拍照→VO→dead-reckoning」的流程跑起來沒有錯誤，靜止拍攝時正確回報位移趨近於 0／`too few inliers`。**尚未實測**：真的接上通電的 Pixhawk、送出 `GPS_INPUT` 並在 Mission Planner 上確認 GPS2 有更新。

### 意外發現的環境問題

`~/.local/lib/python3.10/site-packages` 裡有一個 pip 裝的 `opencv-python`（4.13.0，無 GStreamer），蓋掉了 JetPack 系統內建、有 GStreamer 支援的 OpenCV（`/usr/lib/python3/dist-packages`，4.5.4）。實測確認 **`gpsless_mapping/capture_sample.py` 目前因此無法開啟相機**（`ERROR: Could not open CSI camera`）。`live_vo_gps.py` 用「import cv2 前先把系統路徑插到最前面」繞過去了，但根本問題還沒解決——建議之後把 `~/.local` 那個 opencv-python 移除（前提是沒有其他東西真的需要 4.13 版的特定功能）。

## 檔案清單

| 檔案 | 說明 |
|---|---|
| `config.py` | 相機／演算法／dead-reckoning 全部可調參數 |
| `visual_odometry.py` | 核心 VO 演算法 + CLI |
| `dead_reckoning.py` | 錨點＋dead-reckoning 的 `PositionTracker` |
| `vo_gps_bridge.py` | 批次版：資料夾照片 → VO → GPS_INPUT |
| `live_vo_gps.py` | 即時版：邊拍邊估算邊送 GPS_INPUT |
| `flow.md` | `visual_odometry.py` 流程圖（中文） |
| `photo_obtained/` | 3 張測試照片（含真實 GPS 檔名，僅供驗證） |
| `../gpsless_mapping/gps_sim.py` | 新增 `hacc` 參數，供本專案接入信心標記 |

## 下一步待辦

1. 接上真的 Pixhawk，跑 `live_vo_gps.py`（拿掉 `--dry-run`），在 Mission Planner 上確認 GPS2 有更新、`GPS_AUTO_SWITCH` 行為正常。
2. 用真實飛行 log 校準 `config.py` 裡的 `HACC_GROWTH_FRACTION_*` / `HACC_PENALTY_*`（目前是合理猜測，非實測校準值）。
3. 確認相機安裝角度，校正 `CAMERA_MOUNT_YAW_OFFSET_DEG`（目前預設 0，未針對實機驗證）。
4. 拍攝間隔夠密的真實序列，重新驗證 VO 準確度（目前唯一的實測資料組間隔太遠，準確度不具代表性）。
5. 修掉 `~/.local` 的 opencv-python 蓋掉系統 OpenCV 的問題，讓 `capture_sample.py` 等腳本恢復正常。
6. 串接絕對定位（SIFT／SuperPoint 圖磚比對）成功時自動呼叫 `tracker.reset_anchor()`，取代目前的手動錨點。
