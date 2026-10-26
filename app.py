# ============================================================
# MULTILINGUAL ENTERPRISE AI SUPPORT COPILOT
# STREAMLIT APPLICATION
#
# Classification: Fine-tuned XLM-R
# Retrieval: FAISS + BM25 + RRF
# Generation: Qwen2.5 via Ollama
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

# Import rag_engine first to preserve the working
# PyTorch / FAISS initialization order.

from rag_engine import (
    load_knowledge_base,
    prepare_bm25,
    generate_rag_answer,
)

from ticket_classifier import (
    load_classifier,
    classify_ticket,
)

import streamlit as st


# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Enterprise AI Support Copilot",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Enterprise AI Support Copilot")

st.write(
    "Describe your technical problem in English or Italian. "
    "The assistant will classify the issue, retrieve relevant "
    "Microsoft documentation, and generate troubleshooting steps."
)

st.divider()


# ============================================================
# 3. INITIALIZE RAG
# ============================================================

@st.cache_resource
def initialize_rag():

    index, chunks, embedding_model = load_knowledge_base()

    bm25_data = prepare_bm25(chunks)

    return index, chunks, embedding_model, bm25_data


# ============================================================
# 4. INITIALIZE CLASSIFIER
# ============================================================

@st.cache_resource
def initialize_classifier():

    tokenizer, classifier_model = load_classifier()

    return tokenizer, classifier_model


# ============================================================
# 5. LOAD MODELS
# ============================================================

try:

    with st.spinner("Loading AI models..."):

        index, chunks, embedding_model, bm25_data = initialize_rag()

        tokenizer, classifier_model = initialize_classifier()

except Exception as error:

    st.error("Could not initialize AI models.")

    st.exception(error)

    st.stop()


# ============================================================
# 6. APPLICATION STATUS
# ============================================================

st.success("AI Support Copilot is ready!")

with st.expander("ℹ️ About this AI system"):

    st.markdown(
        """
        **Ticket Classification**
        - Fine-tuned XLM-RoBERTa
        - 18 IT support categories

        **Document Retrieval**
        - FAISS semantic search
        - BM25 keyword search
        - Reciprocal Rank Fusion
        - Classification-aware reranking

        **Answer Generation**
        - Qwen2.5-3B via Ollama
        - English and Italian responses
        - Microsoft documentation sources

        **Note:** AI-generated troubleshooting steps
        should be verified before changing enterprise systems.
        """
    )


# ============================================================
# 7. EMPLOYEE QUESTION
# ============================================================

st.subheader("📝 Describe Your IT Problem")

question = st.text_area(
    "What problem are you experiencing?",
    placeholder=(
        "Example: My VPN connection fails with error 800.\n"
        "Esempio: La connessione Wi-Fi aziendale non funziona."
    ),
    height=140,
)

submit = st.button(
    "🔍 Get Support",
    type="primary",
    use_container_width=True,
)


# ============================================================
# 8. PROCESS QUESTION
# ============================================================

if submit:

    if not question.strip():

        st.warning(
            "Please describe your IT problem."
        )

    else:

        # ----------------------------------------------------
        # STEP 1: CLASSIFICATION
        # ----------------------------------------------------

        with st.spinner("Classifying IT issue..."):

            try:

                classification = classify_ticket(
                    text=question,
                    tokenizer=tokenizer,
                    model=classifier_model,
                )

            except Exception as error:

                st.error("Ticket classification failed.")
                st.exception(error)
                st.stop()

        # ----------------------------------------------------
        # STEP 2: DISPLAY CLASSIFICATION
        # ----------------------------------------------------

        st.divider()

        st.subheader("🏷️ IT Issue Classification")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Predicted Category",
                classification["category"],
            )

        with col2:

            st.metric(
                "Model Confidence",
                f"{classification['confidence'] * 100:.1f}%",
            )

        if classification["confidence"] < 0.60:

            st.info(
                "The classifier has relatively low confidence. "
                "The system will search documentation without "
                "category-based reranking."
            )

        # ----------------------------------------------------
        # STEP 3: RAG GENERATION
        # ----------------------------------------------------

        with st.spinner(
            "Retrieving documentation and generating an answer..."
        ):

            try:
            


                result = generate_rag_answer(
                    question=question,
                    index=index,
                    chunks=chunks,
                    embedding_model=embedding_model,
                    bm25_data=bm25_data,
                    top_k=3,
                    predicted_category=classification["category"],
                    classification_confidence=classification["confidence"],
                )

            except Exception as error:

                st.error("RAG answer generation failed.")
                st.exception(error)
                st.stop()

        # ----------------------------------------------------
        # STEP 4: DISPLAY ANSWER
        # ----------------------------------------------------

        st.success("Your troubleshooting answer is ready!")

        st.subheader("💡 AI Troubleshooting Answer")

        st.markdown(result["answer"])

        # ----------------------------------------------------
        # STEP 5: DISPLAY SOURCES
        # ----------------------------------------------------

        st.divider()

        st.subheader("📚 Retrieved Documentation")

        st.caption(
            "Documentation retrieved using hybrid search "
            "and category-aware reranking."
        )

        if not result["sources"]:

            st.info(
                "No matching documentation was used for this answer."
            )

        for source in result["sources"]:

            chunk = source["chunk"]

            heading = chunk.get(
                "heading",
                "Untitled section",
            )

            with st.expander(
                f"Source {source['rank']}: {heading}"
            ):

                st.write(
                    "**Document:**",
                    chunk.get("document_id", "Unknown"),
                )

                st.write(
                    "**Section:**",
                    heading,
                )

                source_url = chunk.get(
                    "source_url",
                    "",
                )

                if source_url:

                    st.markdown(
                        f"**Original documentation:** "
                        f"[Open source]({source_url})"
                    )

                st.write("**Retrieved content:**")

                st.write(
                    chunk.get("content", "")
                )

                st.caption(
                    f"Hybrid retrieval score: "
                    f"{source['score']:.4f}"
                )

        # ----------------------------------------------------
        # STEP 6: PERFORMANCE
        # ----------------------------------------------------

        st.divider()

        st.subheader("⚡ System Performance")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "RAG Response Time",
                f"{result['response_time']:.2f} s",
            )

        with col2:

            st.metric(
                "Documents Retrieved",
                len(result["sources"]),
            )

        with col3:

            st.metric(
                "Answer Language",
                result["language"],
            )

        # ----------------------------------------------------
        # STEP 7: FEEDBACK
        # ----------------------------------------------------

        st.divider()

        st.subheader("🗣️ Was this answer helpful?")

        feedback = st.radio(
            "Rate the troubleshooting answer:",
            [
                "Helpful",
                "Partially helpful",
                "Not helpful",
            ],
            index=None,
        )

        if feedback:

            st.info(
                f"Feedback selected: {feedback}"
            )

            st.caption(
                "Feedback is not yet stored or used "
                "to retrain the models."
            )