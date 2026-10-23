"""Interface locale : seuls les endpoints FastAPI accèdent au corpus."""
from pathlib import Path
import os
import base64

import httpx
import streamlit as st

st.set_page_config(page_title="MaintiDoc", page_icon="🔧")
st.title("MaintiDoc — assistant documentaire")
st.caption("Texte et photographie • Références documentaires • Vérification par un technicien habilité")
base = os.getenv("MAINTIDOC_API_URL", "http://127.0.0.1:8000")
demo_token = Path(__file__).resolve().parents[1] / "data/runtime/demo-token.txt"
with st.sidebar:
    st.header("Session locale")
    token = st.text_input("Jeton d'accès", type="password", value=demo_token.read_text(encoding="utf-8").strip() if demo_token.exists() else "")
    st.caption("La session préremplie est réservée à cette démonstration sur votre machine.")
    st.caption("Aucun équipement n'est piloté. La création de tickets sera ajoutée à l'étape 3.")
headers = {"Authorization": f"Bearer {token}"}
try:
    response = httpx.get(f"{base}/documents", headers=headers, timeout=10)
    response.raise_for_status()
    documents = response.json()
    st.caption(f"{len(documents)} document(s) accessible(s) à cette session.")
except httpx.HTTPError:
    st.error("API indisponible ou session non autorisée. Vérifiez le démarrage et votre jeton.")
    st.stop()

with st.form("question_form"):
    question = st.text_area("Votre question", placeholder="Quels contrôles sont prévus avant la création d'un ticket ?", max_chars=2000)
    image = st.file_uploader("Photographie facultative — JPEG ou PNG, 5 Mo maximum", type=["jpg", "jpeg", "png"])
    equipment = st.text_input("Référence exacte de l'équipement, si connue", max_chars=120)
    explore_figures = st.checkbox("Explorer les figures et schémas associés")
    submitted = st.form_submit_button("Consulter les documents")
if submitted:
    if len(question.strip()) < 3:
        st.warning("Saisissez une question d'au moins trois caractères.")
    else:
        try:
            with st.spinner("Recherche documentaire et rédaction de la réponse…"):
                if image is not None and image.size > 5 * 1024 * 1024:
                    st.error("La photographie dépasse 5 Mo.")
                    st.stop()
                if explore_figures:
                    response = httpx.post(f"{base}/search-multimodal", headers=headers, json={
                        "question": question,
                        "image_base64": base64.b64encode(image.getvalue()).decode("ascii") if image is not None else None,
                        "equipment_reference": equipment.strip() or None,
                    }, timeout=600)
                elif image is not None:
                    if image.size > 5 * 1024 * 1024:
                        st.error("La photographie dépasse 5 Mo.")
                        st.stop()
                    response = httpx.post(f"{base}/analyze-image", headers=headers, json={
                        "question": question, "image_base64": base64.b64encode(image.getvalue()).decode("ascii"),
                        "equipment_reference": equipment.strip() or None,
                    }, timeout=420)
                else:
                    response = httpx.post(f"{base}/ask", headers=headers, json={"question": question}, timeout=240)
                response.raise_for_status()
            result = response.json()
            if explore_figures:
                st.session_state["references"] = result
                st.session_state.pop("answer", None)
                st.session_state.pop("photo", None)
            else:
                st.session_state.pop("references", None)
                st.session_state["photo"] = result if image is not None else None
                st.session_state["answer"] = result["documentary_answer"] if image is not None else result
            st.session_state["answer_token"] = token
        except httpx.HTTPError as error:
            st.session_state.pop("answer", None)
            st.session_state.pop("photo", None)
            st.session_state.pop("references", None)
            status = error.response.status_code if isinstance(error, httpx.HTTPStatusError) else None
            if status == 409:
                st.error("Les figures doivent être réindexées après la modification du corpus ou des modèles.")
            elif explore_figures:
                st.error("La recherche des figures est indisponible. Vérifiez l'API, les modèles et l'index local.")
            else:
                st.error("La réponse n'a pas pu être obtenue. Vérifiez l'API et Ollama.")
if "answer" in st.session_state and st.session_state.get("answer_token") == token:
    answer = st.session_state["answer"]
    photo = st.session_state.get("photo")
    if photo:
        st.subheader("Observation automatique — à confirmer")
        st.caption("Description en anglais ; elle ne constitue pas un diagnostic.")
        st.text(photo["observation"])
        if photo["clarification"]:
            st.info(photo["clarification"])
        st.caption(" ".join(photo["uncertainties"]))
    if answer["status"] == "abstained" and not (photo and photo["clarification"]):
        st.warning(answer["answer"])
    elif answer["status"] == "answered":
        st.subheader("Réponse")
        st.text(answer["answer"])
    st.caption(answer["notice"])
    fetched = set()
    for index, citation in enumerate(answer["citations"]):
        with st.expander(f"[{citation['label']}] {citation['document']} — page {citation['page']} — version {citation['version']}"):
            st.text(citation["quote"])
            document_id = citation["document_id"]
            if document_id not in fetched:
                fetched.add(document_id)
                try:
                    pdf = httpx.get(f"{base}/documents/{document_id}/pdf", headers=headers, timeout=15)
                    pdf.raise_for_status()
                    st.download_button("Télécharger le PDF pour vérifier", pdf.content,
                                       file_name=citation["document"], mime="application/pdf", key=f"pdf_{index}")
                except httpx.HTTPError:
                    st.warning("PDF inaccessible ou modifié depuis l'indexation.")

if "references" in st.session_state and st.session_state.get("answer_token") == token:
    references = st.session_state["references"]
    st.subheader("Références à vérifier")
    st.caption(references["notice"])
    if references["figures_index_status"] == "not_indexed":
        st.info("Les figures de ce corpus ne sont pas encore indexées.")
    elif references["figures_index_status"] == "empty":
        st.info("L'extraction n'a trouvé aucune région figurative exploitable dans ce corpus.")
    if not references["pages"]:
        st.info("Aucune correspondance accessible pour cette recherche.")
    for page_index, page in enumerate(references["pages"]):
        with st.expander(f"{page['document']} — page {page['page']} — version {page['version']}"):
            for excerpt in page["text_excerpts"]:
                st.text(excerpt)
            for figure in page["figures"]:
                try:
                    image_response = httpx.get(f"{base}/figures/{figure['figure_id']}/image", headers=headers, timeout=15)
                    image_response.raise_for_status()
                    st.image(image_response.content, caption=f"Région extraite de la page {page['page']}")
                except httpx.HTTPError:
                    st.warning("Figure inaccessible ou index à actualiser.")
            try:
                pdf = httpx.get(f"{base}/documents/{page['document_id']}/pdf", headers=headers, timeout=15)
                pdf.raise_for_status()
                st.download_button("Télécharger le PDF source", pdf.content, file_name=page["document"],
                                   mime="application/pdf", key=f"reference_pdf_{page_index}")
            except httpx.HTTPError:
                st.warning("PDF inaccessible ou modifié depuis l'indexation.")
