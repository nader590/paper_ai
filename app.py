import streamlit as st
from transformers import MT5ForConditionalGeneration, MT5Tokenizer, pipeline
import PyPDF2

# تحميل الموديل متعدد اللغات
MODEL_NAME = "google/mt5-small"
tokenizer = MT5Tokenizer.from_pretrained(MODEL_NAME)
model = MT5ForConditionalGeneration.from_pretrained(MODEL_NAME)
generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# دالة لاستخراج النص من PDF
def extract_text_from_pdf(file) -> str:
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text.strip()

# دالة للتوليد من الموديل
def ask_model(prompt: str, max_new_tokens=500):
    outputs = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7
    )
    return outputs[0]["generated_text"]

# واجهة Streamlit
st.title("📄 أداة تحليل الأوراق العلمية")

uploaded_file = st.file_uploader("📥 ارفع ورقة PDF", type=["pdf"])

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)

    st.subheader("🔠 اختر لغة التلخيص")
    lang = st.selectbox("اللغة", ["العربية", "English"])

    if lang == "العربية":
        classify_prompt = f"صنّف هذه الورقة العلمية حسب مجالها ونوعها:\n{text[:1500]}"
        summary_prompt = f"لخّص هذه الورقة العلمية بالعربية في نقاط واضحة:\n{text[:4000]}"
    else:
        classify_prompt = f"Classify this scientific paper by its field and type:\n{text[:1500]}"
        summary_prompt = f"Summarize this scientific paper in clear bullet points in English:\n{text[:4000]}"

    if st.button("🚀 تحليل الورقة"):
        classification = ask_model(classify_prompt, max_new_tokens=200)
        summary = ask_model(summary_prompt, max_new_tokens=400)

        st.write("### 🏷️ التصنيف / Classification")
        st.success(classification)

        st.write("### 📌 الملخص / Summary")
        st.info(summary)

        st.write("### 📖 نص من الورقة (مقتطف)")
        st.text(text[:1000])

# قسم للأسئلة
st.subheader("❓ اسأل عن الورقة")
question = st.text_input("أدخل سؤالك هنا")
context = st.text_area("انسخ النص (أو جزء من الورقة) هنا")

if st.button("💡 احصل على الإجابة"):
    if question and context:
        qa_prompt = f"النص التالي من ورقة علمية:\n{context}\n\nالسؤال: {question}\nالإجابة:"
        answer = ask_model(qa_prompt, max_new_tokens=300)
        st.success(answer)
    else:
        st.warning("⚠️ أدخل سؤالاً ونصاً أولاً")
