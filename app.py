import streamlit as st
from transformers import MT5ForConditionalGeneration, MT5Tokenizer, pipeline
import PyPDF2

# ================= تحميل الموديل =================
@st.cache_resource
def load_model():
    MODEL_NAME = "google/mt5-small"
    tokenizer = MT5Tokenizer.from_pretrained(MODEL_NAME, legacy=False)  # عشان التحذير يختفي
    model = MT5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    return generator

generator = load_model()

# ================= دوال المساعدة =================
def extract_text_from_pdf(file) -> str:
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text.strip()

def ask_model(prompt: str, max_new_tokens=500):
    outputs = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7
    )
    return outputs[0]["generated_text"]

# ================= Streamlit App =================
st.title("📄 AI Paper Assistant (Arabic & English)")

uploaded_file = st.file_uploader("ارفع ورقة PDF", type=["pdf"])

if uploaded_file:
    st.success("✅ تم رفع الملف بنجاح")

    text = extract_text_from_pdf(uploaded_file)
    st.subheader("📑 النص المستخرج من الورقة")
    st.text_area("Extracted Text", text[:3000] + ("..." if len(text) > 3000 else ""), height=200)

    lang = st.radio("🌍 اختر اللغة", ["العربية", "English"])

    if st.button("🔍 تحليل الورقة"):
        with st.spinner("⏳ جاري التحليل..."):

            if lang == "العربية":
                classify_prompt = f"صنّف هذه الورقة العلمية حسب مجالها ونوعها:\n{text[:1500]}"
                summary_prompt = f"لخّص هذه الورقة العلمية بالعربية في نقاط واضحة:\n{text[:2000]}"
            else:
                classify_prompt = f"Classify this research paper by its field and type:\n{text[:1500]}"
                summary_prompt = f"Summarize this research paper in clear points in English:\n{text[:2000]}"

            classification = ask_model(classify_prompt, max_new_tokens=200)
            summary = ask_model(summary_prompt, max_new_tokens=400)

        st.subheader("📌 التصنيف / Classification")
        st.write(classification)

        st.subheader("📝 الملخص / Summary")
        st.write(summary)

    st.subheader("❓ اسأل سؤال عن الورقة")
    question = st.text_input("اكتب سؤالك / Type your question")
    if st.button("إجابة / Answer"):
        if lang == "العربية":
            qa_prompt = f"النص التالي من ورقة علمية:\n{text[:1500]}\n\nالسؤال: {question}\nالإجابة:"
        else:
            qa_prompt = f"The following text is from a research paper:\n{text[:1500]}\n\nQuestion: {question}\nAnswer:"
        
        answer = ask_model(qa_prompt, max_new_tokens=300)
        st.subheader("💡 الإجابة / Answer")
        st.write(answer)
