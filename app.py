import streamlit as st
from transformers import MT5ForConditionalGeneration, MT5Tokenizer, pipeline
import PyPDF2

# ================= تحميل الموديل =================
@st.cache_resource
def load_model():
    MODEL_NAME = "google/mt5-small"
    tokenizer = MT5Tokenizer.from_pretrained(MODEL_NAME)
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
st.title("📄 AI Paper Assistant")

# اختيار اللغة
lang = st.radio("🌍 اختر اللغة / Choose Language", ["العربية", "English"])

# Labels حسب اللغة
labels = {
    "العربية": {
        "upload": "ارفع ورقة PDF",
        "uploaded": "✅ تم رفع الملف بنجاح",
        "extracted": "📑 النص المستخرج من الورقة",
        "analyze": "🔍 تحليل الورقة",
        "classify": "📌 التصنيف",
        "summary": "📝 الملخص",
        "ask": "❓ اسأل سؤال عن الورقة",
        "q_input": "اكتب سؤالك هنا",
        "answer_btn": "إجابة",
        "answer": "💡 الإجابة",
    },
    "English": {
        "upload": "Upload PDF paper",
        "uploaded": "✅ File uploaded successfully",
        "extracted": "📑 Extracted text",
        "analyze": "🔍 Analyze Paper",
        "classify": "📌 Classification",
        "summary": "📝 Summary",
        "ask": "❓ Ask a question about the paper",
        "q_input": "Type your question here",
        "answer_btn": "Answer",
        "answer": "💡 Answer",
    }
}

uploaded_file = st.file_uploader(labels[lang]["upload"], type=["pdf"])

if uploaded_file:
    st.success(labels[lang]["uploaded"])

    text = extract_text_from_pdf(uploaded_file)
    st.subheader(labels[lang]["extracted"])
    st.text_area("Extracted Text", text[:3000] + ("..." if len(text) > 3000 else ""), height=200)

    if st.button(labels[lang]["analyze"]):
        with st.spinner("⏳ Processing..."):

            if lang == "العربية":
                classify_prompt = f"صنّف هذه الورقة العلمية حسب مجالها ونوعها:\n{text[:1500]}"
                summary_prompt = f"لخّص هذه الورقة العلمية بالعربية في نقاط واضحة:\n{text[:2000]}"
            else:
                classify_prompt = f"Classify this research paper by its field and type:\n{text[:1500]}"
                summary_prompt = f"Summarize this research paper in clear points in English:\n{text[:2000]}"

            classification = ask_model(classify_prompt, max_new_tokens=200)
            summary = ask_model(summary_prompt, max_new_tokens=400)

        st.subheader(labels[lang]["classify"])
        st.write(classification)

        st.subheader(labels[lang]["summary"])
        st.write(summary)

    st.subheader(labels[lang]["ask"])
    question = st.text_input(labels[lang]["q_input"])
    if st.button(labels[lang]["answer_btn"]):
        if lang == "العربية":
            qa_prompt = f"النص التالي من ورقة علمية:\n{text[:1500]}\n\nالسؤال: {question}\nالإجابة:"
        else:
            qa_prompt = f"The following text is from a research paper:\n{text[:1500]}\n\nQuestion: {question}\nAnswer:"
        
        answer = ask_model(qa_prompt, max_new_tokens=300)
        st.subheader(labels[lang]["answer"])
        st.write(answer)
