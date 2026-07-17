import os
import gradio as gr
import joblib

# Load the trained model
diabetes_model = joblib.load("diabetes_prediction_model.pkl")


def predict_diabetes(
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    diabetes_pedigree,
    age,
):
    prediction = diabetes_model.predict([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age
    ]])[0]

    if prediction == 1:
        return """
## ⚠️ Positive for Diabetes

The patient may be at risk of diabetes.

**Please consult a healthcare professional for proper diagnosis.**
"""
    else:
        return """
## ✅ No Diabetes Detected

The patient appears to have a low likelihood of diabetes.
"""


css = """
.gradio-container{
    max-width:1000px !important;
    margin:auto;
}

h1{
    text-align:center;
}

.footer{
    text-align:center;
    color:gray;
    font-size:14px;
    margin-top:20px;
}

.result{
    border-radius:12px;
}
"""


with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="emerald"
    ),
    css=css,
    title="Diabetes Prediction System",
) as interface:

    gr.Markdown(
        """
# 🩺 Diabetes Prediction System

### Predict the likelihood of diabetes using Machine Learning

Fill in the patient's medical details below and click **Predict**.
"""
    )

    with gr.Row():

        with gr.Column():

            pregnancies = gr.Number(
                label="👶 Pregnancies",
                value=0
            )

            glucose = gr.Number(
                label="🩸 Glucose Level",
                value=120
            )

            blood_pressure = gr.Number(
                label="💓 Blood Pressure",
                value=70
            )

            skin_thickness = gr.Number(
                label="📏 Skin Thickness",
                value=20
            )

        with gr.Column():

            insulin = gr.Number(
                label="💉 Insulin",
                value=79
            )

            bmi = gr.Number(
                label="⚖️ BMI",
                value=25.0
            )

            diabetes_pedigree = gr.Number(
                label="🧬 Diabetes Pedigree Function",
                value=0.5
            )

            age = gr.Number(
                label="🎂 Age",
                value=30
            )

    predict_btn = gr.Button(
        "🔍 Predict",
        variant="primary",
        size="lg"
    )

    output = gr.Markdown(
        label="Prediction Result",
        elem_classes="result"
    )

    predict_btn.click(
        fn=predict_diabetes,
        inputs=[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree,
            age,
        ],
        outputs=output,
    )

    gr.Markdown(
        """
---
<div class="footer">
Developed with ❤️ using Gradio & Machine Learning
</div>
"""
    )


if __name__ == "__main__":
    interface.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
