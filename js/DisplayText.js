import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

app.registerExtension({
    name: "fnodes.DisplayAny-",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!nodeData?.category?.startsWith("fnodes")) {
            return;
        }
        
        switch (nodeData.name) {
            case "DisplayAny-":
                const onExecutedDisplayAny = nodeType.prototype.onExecuted;
                nodeType.prototype.onExecuted = function (message) {
                    onExecutedDisplayAny?.apply(this, arguments);
                    updateWidget(this, "displaytext", message["text"].join(""));
                };
                break;

            case "ImageScalerForSDModels-":
            case "GetImageSize-":
            case "ImageScaleBySpecifiedSide-":
            case "ComputeImageScaleRatio-":
                const onExecutedImage = nodeType.prototype.onExecuted;
                nodeType.prototype.onExecuted = function (message) {
                    onExecutedImage?.apply(this, arguments);
                    let value = message["width"].join("") + "x" + message["height"].join("");
                    if (nodeData.name === "GetImageSize-") {
                        value += "x" + message["count"].join("");
                    }
                    if (nodeData.name === "ComputeImageScaleRatio-") {
                        value += "x" + message["rescale_ratio"].join("");
                    }
                    updateWidget(this, "return_text", value);
                };
                break;
        }

        // 辅助函数用于更新或创建widget
        function updateWidget(node, widgetName, value) {
            let textWidget = node.widgets && node.widgets.find(w => w.name === widgetName);
            if (!textWidget) {
                textWidget = ComfyWidgets["STRING"](node, widgetName, ["STRING", { multiline: true }], app).widget;
                textWidget.inputEl.readOnly = true;
                textWidget.inputEl.style.border = "none";
                textWidget.inputEl.style.backgroundColor = "transparent";
            }
            textWidget.value = value;
        }
                
    },
});
