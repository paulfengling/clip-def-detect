import argparse
import os
import math

# Gradio's local startup check must bypass system HTTP proxies.
local_hosts = "localhost,127.0.0.1,::1"
for name in ("NO_PROXY", "no_proxy"):
    os.environ[name] = ",".join(filter(None, [os.environ.get(name, ""), local_hosts]))

import gradio as gr

from custom_infer import Predictor, ROOT, render_prediction, inspect_prediction


def build_demo(predictor, class_name="wafer", min_area=25):
    def image_infer(image, object_name, pixel_threshold, area, image_gate):
        if image is None:
            raise gr.Error("Please upload an image.")
        if not object_name or not object_name.strip():
            raise gr.Error("Please enter the product class name.")
        if area is None or isinstance(area, bool) or not math.isfinite(area) or area != int(area) or area < 1:
            raise gr.Error("Minimum defect area must be a positive integer.")
        try:
            anomaly_map, score = predictor.predict(image, object_name.strip())
            overlay, heatmap = render_prediction(image, anomaly_map)
            boxed, mask, result = inspect_prediction(
                predictor, image, anomaly_map, score, object_name,
                pixel_threshold=pixel_threshold, min_area=int(area), image_gate=image_gate)
        except (ValueError, RuntimeError) as error:
            raise gr.Error(str(error)) from error
        return boxed, mask, overlay, heatmap, result

    return gr.Interface(
        fn=image_infer,
        inputs=[gr.Image(type="pil", label="Image / 待检图片"),
                gr.Textbox(value=class_name, label="Product class / 产品英文名"),
                gr.Number(value=None, label="Pixel threshold / 像素阈值（留空读取验证集校准值）"),
                gr.Number(value=min_area, precision=0, label="Minimum defect area / 最小缺陷面积（原图像素）"),
                gr.Checkbox(value=True, label="Suppress boxes on normal images / 正常图片不画框")],
        outputs=[gr.Image(type="pil", label="Defect boxes / 缺陷框"),
                 gr.Image(type="pil", label="Binary defect mask / 二值缺陷区域"),
                 gr.Image(type="pil", label="Defect overlay"), gr.Image(type="pil", label="Heatmap"),
                 gr.JSON(label="Classification and boxes / 判定及框坐标")],
        examples=[[str(path), "bottle", None, min_area, True]
                  for path in (ROOT / "data/opencvxuetang/unknown" / name
                               for name in ("good_bottle.png", "small_bad.png", "big_bad.png"))
                  if class_name == "bottle" and path.is_file()],
        title="AA-CLIP Wafer Defect Inspection", flagging_mode="never", analytics_enabled=False,
        description="先用 wafer 验证集校准模型。热力图颜色仅显示相对强弱；画框使用原始像素阈值。未校准时结果会标记 uncalibrated。缺陷区域分数不是概率。示例图片为已有瓶子数据。",
        concurrency_limit=1,
    )


def main():
    parser = argparse.ArgumentParser(description="AA-CLIP local inference UI")
    parser.add_argument("--save_path", default=str(ROOT / "ckpt/baseline"))
    parser.add_argument("--img_size", type=int)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--class_name", default="wafer")
    parser.add_argument("--min_area", type=int, default=25)
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    predictor = Predictor(args.save_path, args.img_size, args.device)
    build_demo(predictor, args.class_name, args.min_area).launch(server_name="127.0.0.1", server_port=args.port, share=False)


if __name__ == "__main__":
    main()
