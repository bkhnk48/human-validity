<!--
Sync Impact Report
- Version change: 0.1.0 → 0.2.0 (MINOR: mở rộng nguyên tắc II, III, IV và quy trình test)
- Modified principles:
  - II: thêm phạm vi của nhãn Human Validity; gt:<tên> và human cùng lớp positive;
    body trùng / ghép chi / đổi người là thuộc tính chất lượng riêng, không đổi nhãn thành not_human;
    unsure bị loại khỏi tập train và được báo riêng số lượng
  - III: thêm feature contract chung cho train và inference, cấm feature lấy từ GT/annotation
  - IV: chỉ chép phần cần dùng từ repo mẹ, mỗi hàm kế thừa có regression test
- Quy trình làm việc: test liên quan cho mỗi task, toàn bộ suite cho mỗi feature/merge,
  báo số passed/failed/skipped
- Templates: plan/spec/tasks-template.md không cần sửa (đọc constitution lúc chạy)
- TODO:
  - RATIFICATION_DATE: điền khi chủ dự án duyệt bản này.
  - Phiên bản Python: tạm đặt ≥ 3.9 để chạy được code chép từ repo cũ, cần chốt.
  - Công cụ gán nhãn (Streamlit hay HTML tĩnh): để /speckit-plan quyết định.
-->

# Human-Validity Constitution

Dự án gán nhãn và huấn luyện mô hình nhận ra các body sai trong đầu ra của mô hình HPE monocular
(WHAM trước, các mô hình khác sau): body không phải người thật, người thật không có trong GT,
body trùng, body ghép chi của nhiều người, track đổi người. Dữ liệu là Harmony4D và MPI-INF-3DHP.
Repo mẹ: `monocular-hpe-person-detection-and-identity-reliability` (MC-MCF, `pose_assoc/`).

## Core Principles

### I. Nhãn tay là nguồn sự thật
- Nhãn dùng để train và đánh giá là nhãn do người gán. Hungarian, MC-MCF và các luật prelabel
  chỉ đưa ra gợi ý để người duyệt.
- Không lấy nghiệm MC-MCF làm nhãn train. Nghiệm đó phụ thuộc C_FP và α, nên mô hình sẽ học lại
  các thiên lệch ấy.
- Mỗi dòng nhãn ghi người gán và thời điểm gán. Gợi ý tự động phải tách riêng khỏi nhãn đã duyệt.

### II. Quy ước nhãn cố định, có phiên bản
- Nhãn Human Validity chỉ trả lời một câu hỏi: detection có tương ứng với một người thật hay không.
- Đơn vị gán là một đoạn `(body_id, khoảng frame)`, dùng cú pháp `@ frames` như repo mẹ.
- Lớp nhãn: `gt:<tên>`, `human` (người thật không có GT),
  `not_human/{mannequin, chair, reflection, other}`, `unsure`.
  Ảnh người (poster, màn hình, gương) xếp vào `not_human/reflection`.
- `gt:<tên>` và `human` cùng thuộc nhóm positive (người thật). Khi huấn luyện bộ phân loại
  Human Validity, hai nhãn này là cùng một lớp người thật, trừ khi một spec nghiên cứu khác
  quy định rõ.
- Đoạn mang nhãn `unsure` bị loại khỏi tập train. Mỗi thống kê dataset và báo cáo kết quả phải
  ghi riêng số đoạn và số detection `unsure`.
- Frame không có dòng nhãn nghĩa là **chưa duyệt**. Không được hiểu thành "không phải người GT".
- Body trùng (body duplication), body ghép chi (body-part mixing) và đổi người (identity switch)
  là các thuộc tính chất lượng riêng (derived quality attributes), được tính tự động sau khi gán.
  Chúng không phải lớp gán tay và không được dùng để tự động chuyển nhãn Human Validity
  thành `not_human`.
- Mọi thay đổi quy ước phải tăng phiên bản schema nhãn, cập nhật hướng dẫn gán nhãn trong
  `docs/annotation/`, và có script migrate cho nhãn cũ.

### III. Không rò rỉ dữ liệu, không dùng GT lúc suy luận (NON-NEGOTIABLE)
- Chia train/val/test theo take (Harmony4D) và theo subject/sequence (MPI-INF-3DHP), rồi mới gán
  nhãn. Không chia theo camera hay frame, vì các camera cùng một take tương quan rất mạnh.
- Đặc trưng của mô hình chỉ lấy từ đầu ra của mô hình monocular: crop ảnh, keypoint 2D và
  confidence, độ khớp SMPL chiếu với keypoint, tham số SMPL, thống kê track, độ chồng giữa các
  body. GT chỉ dùng để tạo gợi ý nhãn và để đánh giá.
- Lúc train và lúc inference, feature được trích theo cùng một hợp đồng dữ liệu (feature
  contract): cùng danh sách trường đầu vào, cùng một đoạn code. Không được dùng các trường GT,
  kết quả Hungarian, điểm khớp (matching score) với GT hay thông tin annotation làm feature.
  Hợp đồng này phải có test: trích feature chạy được trên một sequence không có file GT nào.

### IV. Tương thích với repo mẹ
- Xuất được `mapping.txt` đúng định dạng cũ để `pose_assoc/` đọc thẳng, không phải sửa repo mẹ.
- Chỉ chép từ repo mẹ phần chức năng thật sự cần (đọc pkl WHAM, đọc GT, phép chiếu
  f = sqrt(W²+H²)), không chép nguyên module lớn như `scripts/wham_gt_mapping.py`.
- Mỗi hàm kế thừa ghi nguồn (file và commit của repo mẹ) và có regression test so kết quả với
  hàm gốc trên dữ liệu mẫu.

### V. Kiểm thử và tái lập được
- Mỗi module trong `src/human_validity/` có test pytest. Hàm `validate` cho nhãn (phủ hết
  detection, không chồng khoảng, tên lớp hợp lệ) phải chạy được trên toàn bộ `data/annotations/`.
- Mọi số liệu đưa vào báo cáo hoặc bài viết phải tái tạo được bằng một lệnh ghi trong README hoặc
  `docs/experiments/`, với seed cố định.
- Độ đồng thuận giữa người gán (Cohen's κ) đo trên ít nhất 20% sequence được gán chồng.

### VI. Đơn giản trước, baseline trước
- Giai đoạn 1 chỉ gồm dữ liệu, gợi ý nhãn, công cụ gán và thống kê. Phần train mô hình làm sau khi
  có dataset v1.
- Trước khi báo cáo một classifier phải có các baseline ngưỡng: confidence của detector,
  confidence keypoint trung bình, độ dài track, NMS/OKS lọc body trùng.
- Không thêm thư viện hoặc tầng trừu tượng khi chưa có nhu cầu cụ thể trong spec.

## Ràng buộc dữ liệu và kỹ thuật

- Python ≥ 3.9 (TODO: chốt). Không dùng cú pháp `X | Y` ở chú thích kiểu được đánh giá lúc chạy.
- Không commit video, ảnh, pkl, checkpoint vào git. Chỉ commit manifest, cấu hình, nhãn dạng
  văn bản (CSV/TXT/YAML) và ảnh ví dụ nhỏ trong hướng dẫn gán nhãn.
- Dữ liệu gốc nằm trên Google Drive và RunPod. Đường dẫn khai báo trong `configs/`, không viết
  cứng trong code.
- Kiểm tra license Harmony4D và MPI-INF-3DHP trước khi phát hành bộ nhãn.

## Quy trình làm việc

- Mỗi tính năng đi theo chuỗi spec-kit: `/speckit-specify` → (`/speckit-clarify`) →
  `/speckit-plan` → `/speckit-tasks` → (`/speckit-analyze`) → `/speckit-implement`.
- Khi đổi định dạng nhãn, CLI hoặc cấu hình: cập nhật README và `docs/` trong cùng commit.
- Trước khi báo xong một task: chạy các test liên quan trực tiếp đến task.
- Trước khi báo xong một feature hoặc merge vào nhánh chính: chạy toàn bộ test suite bằng
  `pytest -q`.
- Mỗi lần chạy test phải báo rõ số test passed, failed và skipped.
- Khi đổi luật prelabel: chạy lại trên ít nhất một sequence đã gán nhãn và so tỉ lệ gợi ý đúng
  với lần trước.

## Governance

- Constitution này đứng trên các quy ước khác của dự án. Mỗi plan phải có mục kiểm tra tuân thủ
  (Constitution Check), và mọi vi phạm phải được giải thích trong mục Complexity Tracking.
- Sửa constitution: ghi lý do, tăng phiên bản theo semver (MAJOR: bỏ hoặc định nghĩa lại nguyên
  tắc; MINOR: thêm nguyên tắc hoặc mở rộng đáng kể; PATCH: sửa câu chữ), chủ dự án duyệt.

**Version**: 0.2.0 | **Ratified**: TODO(RATIFICATION_DATE) | **Last Amended**: 2026-11-18
