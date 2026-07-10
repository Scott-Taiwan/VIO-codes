# `visual_odometry.py` 流程圖

無人機逐張照片位移估算（VO）流程：從 CLI 進入點，到逐對照片的特徵比對、信心度判斷，再到軌跡累加。

**圖例**
- 🟢 信心足夠／繼續往下走
- 🟠 信心不足／提早結束
- ⚪ 僅供驗證用的旁路（不會餵回估算器）

---

## 1. CLI 進入點

### `main(argv)`

解析 `paths` 與可選的 `--altitude`，接著呼叫 `run()`。下面所有步驟都發生在 `run()` 內部。

↓

## 2. 建立照片順序序列

### `list_images(paths)` · `parse_altitude_m(path)`

展開資料夾、依檔案的 `mtime`（拍攝時間）排序；如果沒有給 `--altitude`，就用正規表示式從檔名解析每張照片的高度（例如 `...60.0m.png`）。

↓

## 3. 對每一組相鄰照片 (i, i+1) 執行迴圈

> ### `process_pair(path1, path2, alt1, alt2)`

**3.1 `detect_and_match(gray1, gray2)`**

`_central_crop()` 先裁切兩張影像的中央區域，去掉魚眼鏡頭邊緣的畸變 → `_make_detector()` 選用 SIFT 或 ORB → 對兩張影像各自 `detectAndCompute()` → `BFMatcher.knnMatch(k=2)` → Lowe's ratio test 只留下夠明確、不含糊的配對。回傳的是以「原始整張影像座標」表示的配對點。

- 相關參數：`CENTRAL_CROP_FRACTION`、`DETECTOR`、`MATCH_RATIO`

**3.2 `estimate_similarity(pts1, pts2)`**

`cv2.estimateAffinePartial2D` 搭配 RANSAC，擬合出一個「旋轉＋縮放＋平移」的變換 `M`，並標記哪些配對點符合這個模型（inliers）。

- 相關參數：`RANSAC_REPROJ_THRESHOLD_PX`

↓

**判斷點：`n_inliers < MIN_INLIERS`？**

| 🟠 是 → 提早結束 | 🟢 否 → 繼續 |
|---|---|
| `ok=False`，原因為「inliers 太少」。配對點彼此同意的程度不足以擬合出任何可信的變換，會印出 `[FAIL]`。 | inliers 數量足夠，可以信任 `M` 的形狀，繼續把它換算成實際位移。 |

↓

**`displacement_from_transform(M, altitude_m)`**

把 `M` 拆解成 `scale`（縮放）與世界看起來的旋轉角 `θ`。將影像中心點代入 `M`，中心點「看起來」漂移的方向取負號，就是相機自己的像素位移方向（靜止的地面特徵，看起來會往相機移動的「反方向」流動）。再用 `GSD = altitude / focal_px`（每軸分別計算）換算成公尺。

→ 得到 `dx_m`、`dy_m`、`magnitude_m`、`yaw_change_deg`、`altitude_change_m`（由縮放比例反推）

- 相關參數：`FX_PIXELS`、`FY_PIXELS`

↓

**判斷點：`n_inliers / n_matches < MIN_INLIER_RATIO`？**

重複性高的地面紋理（草地紋路、跑道上重複的白色標記等）可能讓 RANSAC 鎖定一個「彼此自洽但其實是錯的」變換——即使原始配對數量很多也一樣。用 inlier **比例**而非數量，可以抓出這種情況。不論結果如何，`ok=True`、上面算出的位移都會回傳；差別只在信心標記：

- 🟠 `LOW-CONFIDENCE` — 位移值會回傳，但不可信
- 🟢 `OK` — 有信心

- 相關參數：`MIN_INLIER_RATIO`

---

## 4. 回到 `run()` — 累加軌跡

| 🟠 不夠有信心（或失敗） | 🟢 有信心 |
|---|---|
| 這一段不貢獻任何位移，目前位置原地不動地往下傳遞。 | `heading_deg` 累加這一段的 `yaw_change_deg`；`(dx_m, dy_m)` 依這個累積航向角旋轉，換算到「第 0 張照片」的參考座標系後，加進目前累積位置。 |

↓

## 5. CLI 輸出

### `main()` — 印出結果

每一對照片印出 `[OK]` / `[LOW-CONFIDENCE]` / `[FAIL]`、配對與 inlier 數量、位移量、航向角變化、高度變化。最後印出完整的累積軌跡表。

⚪ **僅供驗證，不會餵回估算器：`parse_gps()` + `haversine_m()`**
如果檔名裡帶有真實 GPS 座標，會額外印出真實距離與估算誤差百分比，方便對照。

---

*對應原始碼：`gpsless_VO_image/visual_odometry.py`；上面提到的所有參數都定義在 `config.py`。*
