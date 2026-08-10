# Q3 figure QA

状态：**PASS**；backend：**python**；目标宽度：183 mm。

所有图均由 Python/matplotlib 绘制并导出 PNG、SVG、PDF、TIFF（600 dpi）；PDF 文本最小字号审计阈值为 5 pt。

## q3_industry_shock_heatmap

source rows=12；automated FAIL=0。

Single heatmap inspected at final size: labels fit, cells are legible, no clipping or overlap.

## q3_industry_allocation_shift

source rows=12；automated FAIL=0。

Two panels inspected at final size: signed bars and HHI labels clear; unknown is a separate row; no crop or collision.

## q3_sensitivity_results

source rows=13；automated FAIL=0。

Three panels inspected at final size: explicit human-readable parameter labels, no overlapping error bars, no clipping; each panel has its own scale.
