/* ============================================================
   DocuMind-AI — Frontend Application
   ============================================================ */

const API_BASE = "http://localhost:8000";

// ------------------------------------------------------------------ State
const state = {
    conversationHistory: [],
    isStreaming: false,
    documents: [],
};

// ------------------------------------------------------------------ DOM
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const dom = {
    sidebar: $("#sidebar"),
    sidebarOverlay: $("#sidebarOverlay"),
    mobileMenuBtn: $("#mobileMenuBtn"),
    sidebarToggle: $("#sidebarToggle"),
    uploadZone: $("#uploadZone"),
    fileInput: $("#fileInput"),
    uploadProgress: $("#uploadProgress"),
    progressFill: $("#progressFill"),
    progressText: $("#progressText"),
    docList: $("#docList"),
    docCount: $("#docCount"),
    chatContainer: $("#chatContainer"),
    messages: $("#messages"),
    welcome: $("#welcome"),
    questionInput: $("#questionInput"),
    sendBtn: $("#sendBtn"),
    statusDot: $(".status-dot"),
    statusText: $(".status-text"),
};

// ------------------------------------------------------------------ Init
document.addEventListener("DOMContentLoaded", () => {
    setupEventListeners();
    checkHealth();
    loadDocuments();
});

function setupEventListeners() {
    // Upload
    dom.uploadZone.addEventListener("click", () => dom.fileInput.click());
    dom.fileInput.addEventListener("change", handleFileSelect);
    dom.uploadZone.addEventListener("dragover", handleDragOver);
    dom.uploadZone.addEventListener("dragleave", handleDragLeave);
    dom.uploadZone.addEventListener("drop", handleDrop);

    // Chat
    dom.sendBtn.addEventListener("click", handleSend);
    dom.questionInput.addEventListener("keydown", handleKeyDown);
    dom.questionInput.addEventListener("input", autoResize);

    // Mobile sidebar
    dom.mobileMenuBtn?.addEventListener("click", toggleSidebar);
    dom.sidebarToggle?.addEventListener("click", toggleSidebar);
    dom.sidebarOverlay?.addEventListener("click", closeSidebar);
}

// ================================================================ UPLOAD
function handleDragOver(e) {
    e.preventDefault();
    dom.uploadZone.classList.add("dragover");
}

function handleDragLeave(e) {
    e.preventDefault();
    dom.uploadZone.classList.remove("dragover");
}

function handleDrop(e) {
    e.preventDefault();
    dom.uploadZone.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (file) uploadFile(file);
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) uploadFile(file);
    e.target.value = "";
}

async function uploadFile(file) {
    const allowed = [".pdf", ".docx", ".txt"];
    const ext = "." + file.name.split(".").pop().toLowerCase();
    if (!allowed.includes(ext)) {
        showToast(`Unsupported file: ${ext}. Use PDF, DOCX, or TXT.`, "error");
        return;
    }

    // Show progress
    dom.uploadProgress.classList.add("active");
    dom.progressFill.style.width = "30%";
    dom.progressText.textContent = `Uploading ${file.name}…`;

    const formData = new FormData();
    formData.append("file", file);

    try {
        dom.progressFill.style.width = "60%";
        dom.progressText.textContent = "Processing & indexing…";

        const res = await fetch(`${API_BASE}/upload`, {
            method: "POST",
            body: formData,
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Upload failed");
        }

        const data = await res.json();

        dom.progressFill.style.width = "100%";
        dom.progressText.textContent = `✓ Indexed ${data.chunks_indexed} chunks`;

        showToast(`📄 ${file.name} indexed (${data.chunks_indexed} chunks)`, "success");
        loadDocuments();

        setTimeout(() => {
            dom.uploadProgress.classList.remove("active");
            dom.progressFill.style.width = "0%";
        }, 2000);
    } catch (err) {
        dom.progressText.textContent = `✗ ${err.message}`;
        dom.progressFill.style.width = "100%";
        dom.progressFill.style.background = "var(--error)";
        showToast(err.message, "error");

        setTimeout(() => {
            dom.uploadProgress.classList.remove("active");
            dom.progressFill.style.width = "0%";
            dom.progressFill.style.background = "";
        }, 3000);
    }
}

// ================================================================ DOCUMENTS
async function loadDocuments() {
    try {
        const res = await fetch(`${API_BASE}/documents`);
        if (!res.ok) return;
        const data = await res.json();

        state.documents = data.documents || [];
        renderDocuments();
    } catch {
        // Silently ignore — server may not be running
    }
}

function renderDocuments() {
    dom.docCount.textContent = state.documents.length;

    if (state.documents.length === 0) {
        dom.docList.innerHTML = '<li class="doc-empty">No documents yet</li>';
        return;
    }

    dom.docList.innerHTML = state.documents
        .map((doc) => `<li class="doc-item">${escapeHTML(doc)}</li>`)
        .join("");
}

// ================================================================ CHAT
function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSend();
    }
}

function autoResize() {
    const el = dom.questionInput;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 120) + "px";
}

async function handleSend() {
    const question = dom.questionInput.value.trim();
    if (!question || state.isStreaming) return;

    // Hide welcome
    dom.welcome.classList.add("hidden");

    // Add user message
    addMessage("user", question);
    state.conversationHistory.push({ role: "user", content: question });

    // Clear input
    dom.questionInput.value = "";
    dom.questionInput.style.height = "auto";
    dom.sendBtn.disabled = true;
    state.isStreaming = true;

    // Add assistant placeholder with typing indicator
    const assistantId = addMessage("assistant", null, true);

    try {
        const res = await fetch(`${API_BASE}/ask/stream`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                question,
                history: state.conversationHistory.slice(-6),
            }),
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Failed to get answer");
        }

        // Stream response
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let fullAnswer = "";

        removeTypingIndicator(assistantId);

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split("\n");

            for (const line of lines) {
                if (line.startsWith("data: ")) {
                    const data = line.slice(6);  // DON'T trim — spaces are meaningful tokens
                    if (data.trim() === "[DONE]") continue;
                    if (data.trim().startsWith("[ERROR]")) {
                        throw new Error(data.trim().slice(8));
                    }
                    // Decode escaped newlines from SSE
                    const decoded = data.replace(/\\n/g, "\n");
                    fullAnswer += decoded;
                    updateMessage(assistantId, formatAnswer(fullAnswer));
                }
            }
        }

        state.conversationHistory.push({ role: "assistant", content: fullAnswer });
    } catch (err) {
        removeTypingIndicator(assistantId);
        updateMessage(
            assistantId,
            `<span style="color: var(--error)">⚠ ${escapeHTML(err.message)}</span>`
        );
    } finally {
        state.isStreaming = false;
        dom.sendBtn.disabled = false;
        dom.questionInput.focus();
    }
}

// Suggestion buttons
window.askSuggestion = function (text) {
    dom.questionInput.value = text;
    handleSend();
};

// ================================================================ MESSAGES
let messageCounter = 0;

function addMessage(role, content, typing = false) {
    const id = `msg-${++messageCounter}`;
    const avatar = role === "user" ? "👤" : "🧠";

    const html = `
        <div class="message ${role}" id="${id}">
            <div class="message-avatar">${avatar}</div>
            <div class="message-content">
                ${typing ? typingIndicatorHTML() : ""}
                ${content ? `<div class="message-text">${formatAnswer(content)}</div>` : '<div class="message-text"></div>'}
            </div>
        </div>
    `;

    dom.messages.insertAdjacentHTML("beforeend", html);
    scrollToBottom();
    return id;
}

function updateMessage(id, html) {
    const el = document.querySelector(`#${id} .message-text`);
    if (el) {
        el.innerHTML = html;
        scrollToBottom();
    }
}

function removeTypingIndicator(id) {
    const el = document.querySelector(`#${id} .typing-indicator`);
    if (el) el.remove();
}

function typingIndicatorHTML() {
    return `<div class="typing-indicator"><span></span><span></span><span></span></div>`;
}

function scrollToBottom() {
    dom.chatContainer.scrollTop = dom.chatContainer.scrollHeight;
}

// ================================================================ FORMAT
function formatAnswer(text) {
    // Very basic markdown-ish rendering
    let html = escapeHTML(text);

    // Bold
    html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
    // Italic
    html = html.replace(/\*(.*?)\*/g, "<em>$1</em>");
    // Inline code
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
    // Line breaks
    html = html.replace(/\n/g, "<br>");

    return html;
}

function escapeHTML(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}

// ================================================================ HEALTH
async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE}/health`);
        if (res.ok) {
            dom.statusDot.classList.add("online");
            dom.statusDot.classList.remove("offline");
            dom.statusText.textContent = "Server online";
        } else {
            throw new Error();
        }
    } catch {
        dom.statusDot.classList.add("offline");
        dom.statusDot.classList.remove("online");
        dom.statusText.textContent = "Server offline";
    }
}

// Poll health every 30s
setInterval(checkHealth, 30000);

// ================================================================ SIDEBAR
function toggleSidebar() {
    dom.sidebar.classList.toggle("open");
    dom.sidebarOverlay.classList.toggle("active");
}

function closeSidebar() {
    dom.sidebar.classList.remove("open");
    dom.sidebarOverlay.classList.remove("active");
}

// ================================================================ TOAST
function showToast(message, type = "info") {
    const toast = document.createElement("div");
    toast.style.cssText = `
        position: fixed; bottom: 24px; right: 24px; z-index: 10000;
        padding: 12px 20px; border-radius: 12px;
        font-family: var(--font); font-size: 0.9rem;
        color: #fff; max-width: 350px;
        animation: messageIn 0.3s ease;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        ${type === "error"
            ? "background: linear-gradient(135deg, #e17055, #d63031);"
            : "background: linear-gradient(135deg, #00b894, #00cec9);"
        }
    `;
    toast.textContent = message;
    document.body.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = "0";
        toast.style.transition = "opacity 0.3s ease";
        setTimeout(() => toast.remove(), 300);
    }, 3500);
}
