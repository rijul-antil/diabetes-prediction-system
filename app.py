import os
import joblib
import gradio as gr

# Load Model
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

    values = [
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age,
    ]

    # Validation
    if any(v < 0 for v in values):
        return "❌ Please enter valid (non-negative) values."

    prediction = diabetes_model.predict([values])[0]

    if prediction == 1:
        return """
## ⚠️ Diabetes Detected

The patient may have diabetes.

### Recommendation
- Consult a doctor.
- Exercise regularly.
- Reduce sugar intake.
- Maintain a healthy lifestyle.
"""
    else:
        return """
## ✅ No Diabetes Detected

The patient appears to have a low risk of diabetes.

### Recommendation
- Continue a healthy lifestyle.
- Exercise regularly.
- Eat a balanced diet.
"""


css = """
.gradio-container{
    max-width:950px;
    margin:auto;
}

h1{
    text-align:center;
}

.footer{
    text-align:center;
    color:gray;
    margin-top:20px;
    font-size:14px;
}

.result{
    border:2px solid #4CAF50;
    border-radius:10px;
    padding:15px;
}
"""


with gr.Blocks(
    theme=gr.themes.Soft(),
    css=css,
    title="Diabetes Prediction System"
) as interface:

    gr.Markdown("""
# 🩺 Diabetes Prediction System

### Machine Learning Based Diabetes Prediction

---

### 👨‍💻 Developer Details

**Name:** Rijul

**Roll No.:** 28241282

**Institute:** Panipat Institute of Engineering and Technology

---

Fill all patient details and click **Predict**.
""")

    with gr.Row():

        with gr.Column():

            pregnancies = gr.Number(label="👶 Pregnancies", value=0)
            glucose = gr.Number(label="🩸 Glucose", value=120)
            blood_pressure = gr.Number(label="💓 Blood Pressure", value=70)
            skin_thickness = gr.Number(label="📏 Skin Thickness", value=20)

        with gr.Column():

            insulin = gr.Number(label="💉 Insulin", value=79)
            bmi = gr.Number(label="⚖️ BMI", value=25.0)
            diabetes_pedigree = gr.Number(
                label="🧬 Diabetes Pedigree Function",
                value=0.5
            )
            age = gr.Number(label="🎂 Age", value=30)

    with gr.Row():

        predict_btn = gr.Button("🔍 Predict", variant="primary")
        clear_btn = gr.ClearButton()

    output = gr.Markdown(elem_classes="result")

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

    gr.Markdown("""
---

### 📢 Disclaimer

This application is developed for educational purposes only.
It should not be used as a substitute for professional medical advice.

---

<div class="footer">

❤️ Developed by <b>Rijul</b><br>

Roll No.: <b>28241282</b><br>

Panipat Institute of Engineering and Technology

</div>
""")

if __name__ == "__main__":
    interface.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
