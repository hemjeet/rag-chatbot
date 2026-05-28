"""
Gradio frontend for the RAG system.
Designed to be mounted inside the FastAPI app via `gr.mount_gradio_app`.
Communicates directly with the RAGEngine instance (no HTTP round-trips).
"""

from __future__ import annotations

import uuid
import logging
from typing import Optional, List, Tuple

import gradio as gr
from memory import memory_manager

logger = logging.getLogger("gradio_app")

# Type alias for Gradio chatbot history
ChatHistory = List[dict]


# ──────────────────────────────────────────────
# Custom Theme / CSS
# ──────────────────────────────────────────────

CUSTOM_CSS = """
/* ── Global ── */
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}

/* ── Header ── */
#header-row {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 16px;
}
#header-row * {
    color: #ffffff !important;
}

/* ── Chatbot styling ── */
.chatbot-container .message {
    border-radius: 12px !important;
}

/* ── File upload area ── */
.upload-area {
    border: 2px dashed #6c63ff !important;
    border-radius: 12px !important;
    background: rgba(108, 99, 255, 0.04) !important;
    transition: all 0.3s ease;
}
.upload-area:hover {
    border-color: #8b83ff !important;
    background: rgba(108, 99, 255, 0.08) !important;
}

/* ── Status badges ── */
.status-ready {
    background: linear-gradient(135deg, #00c853, #64dd17);
    color: white;
    padding: 6px 16px;
    border-radius: 20px;
    font-weight: 600;
    display: inline-block;
}
.status-waiting {
    background: linear-gradient(135deg, #ff6d00, #ff9100);
    color: white;
    padding: 6px 16px;
    border-radius: 20px;
    font-weight: 600;
    display: inline-block;
}

/* ── Cards ── */
.settings-card {
    background: rgba(108, 99, 255, 0.05);
    border: 1px solid rgba(108, 99, 255, 0.15);
    border-radius: 12px;
    padding: 20px;
}

/* ── Buttons ── */
.primary-btn {
    background: linear-gradient(135deg, #6c63ff 0%, #8b83ff 100%) !important;
    border: none !important;
    color: white !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}
.primary-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(108, 99, 255, 0.4) !important;
}
.danger-btn {
    background: linear-gradient(135deg, #ff1744 0%, #ff5252 100%) !important;
    border: none !important;
    color: white !important;
    border-radius: 10px !important;
}
"""


def build_interface(engine) -> gr.Blocks:
    """
    Build the Gradio Blocks interface.
    Receives the RAGEngine instance directly (no HTTP needed).
    """

    theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.indigo,
        secondary_hue=gr.themes.colors.purple,
        neutral_hue=gr.themes.colors.gray,
        font=gr.themes.GoogleFont("Inter"),
    ).set(
        body_background_fill="*neutral_50",
        block_background_fill="white",
        block_border_width="0px",
        block_shadow="0 1px 3px rgba(0,0,0,0.08)",
        button_primary_background_fill="linear-gradient(135deg, #6c63ff 0%, #8b83ff 100%)",
        button_primary_text_color="white",
        input_border_width="1px",
        input_border_color="*neutral_200",
        input_background_fill="white",
    )

    with gr.Blocks(
        theme=theme,
        css=CUSTOM_CSS,
        title="RAG System — Chat with your Documents",
    ) as demo:

        # ── State ──
        session_id = gr.State(value=lambda: str(uuid.uuid4()))

        # ══════════════════════════════════════
        # HEADER
        # ══════════════════════════════════════
        with gr.Row(elem_id="header-row"):
            with gr.Column():
                gr.Markdown(
                    "# 📚 RAG System\n"
                    "### Chat with your documents using AI-powered retrieval"
                )

        # ══════════════════════════════════════
        # TABS
        # ══════════════════════════════════════
        with gr.Tabs() as tabs:

            # ── TAB 1: Chat ──
            with gr.Tab("💬 Chat", id="chat-tab"):
                with gr.Row():
                    with gr.Column(scale=4):
                        chatbot = gr.Chatbot(
                            label="Conversation",
                            height=520,
                            avatar_images=(None, "https://cdn-icons-png.flaticon.com/512/4712/4712027.png"),
                            type="messages",
                            placeholder=(
                                "<h3 style='text-align:center; color:#6c63ff;'>"
                                "Upload documents & start asking questions!</h3>"
                            ),
                        )

                        with gr.Row(equal_height=True):
                            msg_input = gr.Textbox(
                                placeholder="Ask a question about your documents...",
                                show_label=False,
                                scale=6,
                                autofocus=True,
                            )
                            send_btn = gr.Button(
                                "Send ➤",
                                variant="primary",
                                scale=1,
                                elem_classes=["primary-btn"],
                            )

                        with gr.Row():
                            clear_chat_btn = gr.Button("🗑️ Clear Chat", size="sm")
                            rerank_toggle = gr.Checkbox(
                                label="⚡ Enable Reranking (NVIDIA)",
                                value=False,
                                interactive=True,
                            )

                    # Sidebar — source info
                    with gr.Column(scale=1, min_width=220):
                        gr.Markdown("### 📋 Status")
                        status_display = gr.Markdown(
                            _render_status(engine),
                            every=None,
                        )
                        refresh_status_btn = gr.Button("🔄 Refresh", size="sm")

                        gr.Markdown("---")
                        gr.Markdown("### 📄 Sources")
                        sources_display = gr.Markdown("_Ask a question to see sources_")

            # ── TAB 2: Documents ──
            with gr.Tab("📁 Documents", id="docs-tab"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### Upload Documents")
                        gr.Markdown(
                            "Supported formats: **PDF**, **TXT**, **DOCX**, **Markdown**, **CSV**, **HTML**"
                        )
                        file_upload = gr.File(
                            label="Drop files here or click to upload",
                            file_count="multiple",
                            file_types=[".txt", ".pdf", ".docx", ".md", ".csv", ".html", ".htm"],
                            elem_classes=["upload-area"],
                        )
                        upload_btn = gr.Button(
                            "📥 Process & Index Documents",
                            variant="primary",
                            elem_classes=["primary-btn"],
                        )
                        upload_status = gr.Markdown("")

                    with gr.Column(scale=1):
                        gr.Markdown("### Indexed Documents")
                        docs_list = gr.Markdown(_render_docs_list(engine))
                        refresh_docs_btn = gr.Button("🔄 Refresh List", size="sm")
                        clear_kb_btn = gr.Button(
                            "🗑️ Clear Knowledge Base",
                            variant="stop",
                            elem_classes=["danger-btn"],
                        )
                        clear_kb_status = gr.Markdown("")

            # ── TAB 3: Settings ──
            with gr.Tab("⚙️ Settings", id="settings-tab"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🔑 API Keys")
                        with gr.Group(elem_classes=["settings-card"]):
                            openai_key_input = gr.Textbox(
                                label="OpenAI API Key",
                                type="password",
                                placeholder="sk-...",
                                value=settings.openai_api_key if settings.openai_api_key else "",
                            )
                            deepseek_key_input = gr.Textbox(
                                label="DeepSeek API Key",
                                type="password",
                                placeholder="sk-...",
                                value=settings.deepseek_api_key if settings.deepseek_api_key else "",
                            )
                            nvidia_key_input = gr.Textbox(
                                label="NVIDIA API Key (for reranking)",
                                type="password",
                                placeholder="nvapi-...",
                                value=settings.nvidia_api_key if settings.nvidia_api_key else "",
                            )

                    with gr.Column():
                        gr.Markdown("### 🤖 Model Configuration")
                        with gr.Group(elem_classes=["settings-card"]):
                            provider_dropdown = gr.Dropdown(
                                label="LLM Provider",
                                choices=["openai", "deepseek"],
                                value=engine.get_config()["provider"] if engine else "openai",
                            )
                            model_dropdown = gr.Dropdown(
                                label="Model",
                                choices=_get_model_choices(
                                    engine.get_config()["provider"] if engine else "openai"
                                ),
                                value=engine.get_config()["model"] if engine else "gpt-4o",
                            )
                            embedding_dropdown = gr.Dropdown(
                                label="Embedding Model",
                                choices=[
                                    "text-embedding-3-small",
                                    "text-embedding-3-large",
                                    "text-embedding-ada-002",
                                ],
                                value=(
                                    engine.get_config()["embedding_model"]
                                    if engine else "text-embedding-3-small"
                                ),
                            )
                            temperature_slider = gr.Slider(
                                label="Temperature",
                                minimum=0.0,
                                maximum=1.0,
                                step=0.1,
                                value=engine.get_config()["temperature"] if engine else 0.0,
                            )
                            save_settings_btn = gr.Button(
                                "💾 Save Settings",
                                variant="primary",
                                elem_classes=["primary-btn"],
                            )
                            settings_status = gr.Markdown("")

        # ══════════════════════════════════════
        # EVENT HANDLERS
        # ══════════════════════════════════════

        # ── Chat: Send message ──
        async def handle_message(
            message: str,
            history: ChatHistory,
            sid: str,
            rerank: bool,
        ):
            if not message.strip():
                yield history, "", "_Type a question first_"
                return

            if not engine or not engine.is_ready:
                history = history + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": "⚠️ No documents indexed yet. Please upload and process documents first in the **Documents** tab."},
                ]
                yield history, "", ""
                return

            # Add user message to history
            history = history + [{"role": "user", "content": message}]
            # Add empty assistant message for streaming
            history = history + [{"role": "assistant", "content": ""}]

            yield history, "", "_Searching..._"

            # Stream the response
            full_answer = ""
            sources_text = "_No sources_"
            try:
                import json

                async for token in engine.query_stream(
                    question=message,
                    session_id=sid,
                    rerank=rerank,
                ):
                    # Intercept the sources JSON marker
                    if token.startswith('{"__sources__"'):
                        try:
                            data = json.loads(token)
                            src_list = data["__sources__"]
                            if src_list:
                                sources_text = "**Sources used:**\n" + "\n".join(f"- 📄 `{s}`" for s in src_list)
                            else:
                                sources_text = "_No direct sources found_"
                            
                            # Update UI immediately with sources before generating answer
                            yield history, "", sources_text
                            continue
                        except:
                            pass
                    
                    full_answer += token
                    # Update the last message in history with the newly accumulated answer
                    history[-1] = {"role": "assistant", "content": full_answer}
                    
                    # Yield back to Gradio to render the new token on screen
                    yield history, "", sources_text

            except Exception as e:
                logger.exception("Chat error")
                history[-1] = {"role": "assistant", "content": f"❌ Error: {e}"}
                yield history, "", ""

        send_btn.click(
            fn=handle_message,
            inputs=[msg_input, chatbot, session_id, rerank_toggle],
            outputs=[chatbot, msg_input, sources_display],
        )
        msg_input.submit(
            fn=handle_message,
            inputs=[msg_input, chatbot, session_id, rerank_toggle],
            outputs=[chatbot, msg_input, sources_display],
        )

        # ── Chat: Clear ──
        def clear_chat(sid: str):
            memory_manager.delete_session(sid)
            new_sid = str(uuid.uuid4())
            return [], new_sid, "_Ask a question to see sources_"

        clear_chat_btn.click(
            fn=clear_chat,
            inputs=[session_id],
            outputs=[chatbot, session_id, sources_display],
        )

        # ── Status refresh ──
        def refresh_status():
            return _render_status(engine)

        refresh_status_btn.click(fn=refresh_status, outputs=[status_display])

        # ── Documents: Upload ──
        def handle_upload(files):
            if not files:
                return "⚠️ No files selected.", _render_docs_list(engine)

            if not engine:
                return "❌ Engine not initialized.", _render_docs_list(engine)

            try:
                file_paths = [f.name for f in files]
                result = engine.ingest_documents(file_paths)
                status = (
                    f"✅ **Success!** Processed {len(result['files_processed'])} file(s).\n\n"
                    f"- New chunks: **{result['new_chunks']}**\n"
                    f"- Total chunks: **{result['total_chunks']}**\n"
                    f"- Total files: **{result['total_files']}**"
                )
                return status, _render_docs_list(engine)
            except Exception as e:
                logger.exception("Upload error")
                return f"❌ Error: {e}", _render_docs_list(engine)

        upload_btn.click(
            fn=handle_upload,
            inputs=[file_upload],
            outputs=[upload_status, docs_list],
        )

        # ── Documents: Refresh list ──
        refresh_docs_btn.click(
            fn=lambda: _render_docs_list(engine),
            outputs=[docs_list],
        )

        # ── Documents: Clear KB ──
        def handle_clear_kb():
            if not engine:
                return "❌ Engine not initialized.", _render_docs_list(engine)
            result = engine.clear_index()
            return f"✅ {result['message']}", _render_docs_list(engine)

        clear_kb_btn.click(
            fn=handle_clear_kb,
            outputs=[clear_kb_status, docs_list],
        )

        # ── Settings: Update model choices on provider change ──
        def on_provider_change(provider: str):
            choices = _get_model_choices(provider)
            return gr.update(choices=choices, value=choices[0] if choices else "")

        provider_dropdown.change(
            fn=on_provider_change,
            inputs=[provider_dropdown],
            outputs=[model_dropdown],
        )

        # ── Settings: Save ──
        def save_settings(
            openai_key, deepseek_key, nvidia_key,
            provider, model, embedding, temp,
        ):
            if not engine:
                return "❌ Engine not initialized."
            try:
                engine.update_config(
                    provider=provider,
                    model=model,
                    embedding_model=embedding,
                    temperature=temp,
                    openai_api_key=openai_key if openai_key else None,
                    deepseek_api_key=deepseek_key if deepseek_key else None,
                    nvidia_api_key=nvidia_key if nvidia_key else None,
                )
                return "✅ Settings saved successfully!"
            except Exception as e:
                return f"❌ Error saving settings: {e}"

        save_settings_btn.click(
            fn=save_settings,
            inputs=[
                openai_key_input, deepseek_key_input, nvidia_key_input,
                provider_dropdown, model_dropdown, embedding_dropdown,
                temperature_slider,
            ],
            outputs=[settings_status],
        )

    return demo


# ──────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────

from config import settings, LLM_PROVIDERS


def _get_model_choices(provider: str) -> list:
    """Get available model names for a provider."""
    provider_info = LLM_PROVIDERS.get(provider, {})
    return list(provider_info.get("models", {}).keys())


def _render_status(engine) -> str:
    """Render the status sidebar widget."""
    if not engine:
        return '<div class="status-waiting">⏳ Initializing...</div>'

    config = engine.get_config()
    if config["is_ready"]:
        return (
            f'<div class="status-ready">✅ Ready</div>\n\n'
            f"**Provider:** {config['provider']}\n\n"
            f"**Model:** {config['model']}\n\n"
            f"**Files:** {len(config['indexed_files'])}\n\n"
            f"**Chunks:** {config['total_chunks']}"
        )
    else:
        return (
            '<div class="status-waiting">📭 No documents</div>\n\n'
            "Upload documents in the **Documents** tab to get started."
        )


def _render_docs_list(engine) -> str:
    """Render the indexed documents list."""
    if not engine:
        return "_Engine not initialized_"

    docs = engine.list_documents()
    if not docs["files"]:
        return "_No documents indexed yet_"

    lines = [f"**{len(docs['files'])} file(s)** — {docs['total_chunks']} chunks\n"]
    for i, f in enumerate(docs["files"], 1):
        lines.append(f"{i}. 📄 `{f}`")
    return "\n".join(lines)


def _format_sources(engine) -> str:
    """Format source files for display."""
    if not engine:
        return ""
    docs = engine.list_documents()
    if not docs["files"]:
        return "_No sources_"

    lines = ["**Indexed sources:**"]
    for f in docs["files"]:
        lines.append(f"- 📄 `{f}`")
    return "\n".join(lines)
