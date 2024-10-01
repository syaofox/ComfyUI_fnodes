import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

app.registerExtension({
    name: "fnodes.DisplayAny-",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!nodeData?.category?.startsWith("fnodes")) {
            return;
        }
        
        if (nodeData.name === "DisplayAny-" )  {
            const onExecuted = nodeType.prototype.onExecuted;

            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);

                let textWidget = this.widgets && this.widgets.find(w => w.name === "displaytext");
                if (!textWidget) {
                    textWidget = ComfyWidgets["STRING"](this, "displaytext", ["STRING", { multiline: true }], app).widget;
                    textWidget.inputEl.readOnly = true;
                    textWidget.inputEl.style.border = "none";
                    textWidget.inputEl.style.backgroundColor = "transparent";
                }
                textWidget.value = message["text"].join("");
            };

        };

        if (nodeData.name === "ImageScalerForSDModels-" || nodeData.name === "GetImageSize-" || nodeData.name === "ImageScaleBySpecifiedSide-" || nodeData.name === "ComputeImageScaleRatio-") {
            const onExecuted = nodeType.prototype.onExecuted;

            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);

                let textWidget = this.widgets && this.widgets.find(w => w.name === "return_text");
                if (!textWidget) {
                    textWidget = ComfyWidgets["STRING"](this, "return_text", ["STRING", { multiline: true }], app).widget;
                    textWidget.inputEl.readOnly = true;
                    textWidget.inputEl.style.border = "none";
                    textWidget.inputEl.style.backgroundColor = "transparent";
                }
                textWidget.value = message["width"].join("") + "x" + message["height"].join("");
                if (nodeData.name === "GetImageSize-") {
                    textWidget.value += "x" + message["count"].join("");
                }
                if (nodeData.name === "ComputeImageScaleRatio-") {
                    textWidget.value += "x" + message["rescale_ratio"].join("");
                }
            };

        }
                
    },
});
