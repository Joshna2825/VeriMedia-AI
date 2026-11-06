// CapCheck - Interactive Client Application Logic

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("file-input");
    const progressContainer = document.getElementById("upload-progress-container");
    const statusText = document.getElementById("analysis-status-text");
    const resultsSection = document.getElementById("results-section");

    // Verdict Elements
    const mediaTypeBadge = document.getElementById("media-type-badge");
    const verdictBadge = document.getElementById("verdict-badge");
    const aiProbNum = document.getElementById("ai-prob-num");
    const realProbNum = document.getElementById("real-prob-num");
    const aiProbBar = document.getElementById("ai-prob-bar");
    const realProbBar = document.getElementById("real-prob-bar");
    const confidenceBadge = document.getElementById("confidence-badge");
    const vivaPitchText = document.getElementById("viva-pitch-text");
    const previewContainer = document.getElementById("media-preview-container");
    const previewFilename = document.getElementById("preview-filename");

    // Visualizers
    const imgVisualizer = document.getElementById("image-visualizer-container");
    const videoVisualizer = document.getElementById("video-visualizer-container");
    const forensicCardsList = document.getElementById("forensic-cards-list");

    // Image Views
    const elaOrigView = document.getElementById("ela-orig-view");
    const elaDiffView = document.getElementById("ela-diff-view");
    const fftOrigView = document.getElementById("fft-orig-view");
    const fftSpecView = document.getElementById("fft-spec-view");
    const noiseOrigView = document.getElementById("noise-orig-view");
    const noiseResView = document.getElementById("noise-res-view");

    // Video Views
    const vstatSsim = document.getElementById("vstat-ssim");
    const vstatJitter = document.getElementById("vstat-jitter");
    const vstatFrames = document.getElementById("vstat-frames");
    const vstatDuration = document.getElementById("vstat-duration");
    const temporalCanvas = document.getElementById("temporal-chart");
    const kfHighestImg = document.getElementById("kf-highest-img");
    const kfHighestTime = document.getElementById("kf-highest-time");
    const kfHighestScore = document.getElementById("kf-highest-score");
    const kfLowestImg = document.getElementById("kf-lowest-img");
    const kfLowestTime = document.getElementById("kf-lowest-time");
    const kfLowestScore = document.getElementById("kf-lowest-score");

    // Modals
    const vivaModal = document.getElementById("viva-modal");
    const archModal = document.getElementById("arch-modal");
    const btnOpenViva = document.getElementById("btn-open-viva-modal");
    const btnCloseViva = document.getElementById("btn-close-viva");
    const btnOpenArch = document.getElementById("btn-open-arch-modal");
    const btnCloseArch = document.getElementById("btn-close-arch");
    const btnPrintReport = document.getElementById("btn-print-report");

    // -------------------------------------------------------------
    // Dropzone Events
    // -------------------------------------------------------------
    dropzone.addEventListener("click", () => fileInput.click());

    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("dragover");
    });

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    // Quick Samples Buttons
    document.querySelectorAll(".sample-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
            const sampleId = btn.getAttribute("data-sample");
            handleSampleSelect(sampleId);
        });
    });

    // Tab Switching
    document.querySelectorAll(".tab-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
            document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
            document.querySelectorAll(".tab-pane").forEach((p) => p.classList.remove("active"));

            btn.classList.add("active");
            const targetId = btn.getAttribute("data-tab");
            const targetPane = document.getElementById(targetId);
            if (targetPane) targetPane.classList.add("active");
        });
    });

    // Modal Events
    btnOpenViva.addEventListener("click", () => vivaModal.classList.remove("hidden"));
    btnCloseViva.addEventListener("click", () => vivaModal.classList.add("hidden"));
    btnOpenArch.addEventListener("click", () => archModal.classList.remove("hidden"));
    btnCloseArch.addEventListener("click", () => archModal.classList.add("hidden"));

    window.addEventListener("click", (e) => {
        if (e.target === vivaModal) vivaModal.classList.add("hidden");
        if (e.target === archModal) archModal.classList.add("hidden");
    });

    btnPrintReport.addEventListener("click", () => window.print());

    // -------------------------------------------------------------
    // Upload & Analysis Handlers
    // -------------------------------------------------------------
    function setLoading(isLoading, text = "Analyzing multi-pillar forensic signals...") {
        if (isLoading) {
            progressContainer.classList.remove("hidden");
            statusText.textContent = text;
        } else {
            progressContainer.classList.add("hidden");
        }
    }

    async function handleFileUpload(file) {
        setLoading(true, `Uploading & analyzing ${file.name}...`);
        const formData = new FormData();
        formData.append("file", file);

        try {
            const resp = await fetch("/api/analyze", {
                method: "POST",
                body: formData,
            });
            const data = await resp.json();
            if (data.success) {
                // If it's an image, create an object URL for preview
                let objectUrl = "";
                if (data.media_type === "image") {
                    objectUrl = URL.createObjectURL(file);
                }
                renderResults(data, objectUrl);
            } else {
                alert("Analysis error: " + (data.error || "Unknown error"));
            }
        } catch (err) {
            alert("Network error: " + err.message);
        } finally {
            setLoading(false);
        }
    }

    async function handleSampleSelect(sampleId) {
        setLoading(true, `Loading & evaluating sample (${sampleId})...`);
        try {
            const resp = await fetch(`/api/analyze-sample/${sampleId}`, {
                method: "POST",
            });
            const data = await resp.json();
            if (data.success) {
                renderResults(data, data.sample_media_url);
            } else {
                alert("Sample error: " + (data.error || "Failed to load sample"));
            }
        } catch (err) {
            alert("Network error: " + err.message);
        } finally {
            setLoading(false);
        }
    }

    // -------------------------------------------------------------
    // Results Rendering
    // -------------------------------------------------------------
    function renderResults(data, mediaUrl) {
        resultsSection.classList.remove("hidden");
        resultsSection.scrollIntoView({ behavior: "smooth" });

        const verdict = data.verdict;
        const isVideo = data.media_type === "video";

        // Badges & Header
        mediaTypeBadge.textContent = isVideo ? "VIDEO" : "IMAGE";
        verdictBadge.textContent = verdict.verdict;
        verdictBadge.className = `badge ${verdict.verdict_badge}`;

        // Percentage Gauges
        aiProbNum.textContent = `${verdict.ai_probability}%`;
        realProbNum.textContent = `${verdict.real_probability}%`;
        aiProbBar.style.width = `${verdict.ai_probability}%`;
        realProbBar.style.width = `${verdict.real_probability}%`;

        confidenceBadge.textContent = verdict.confidence;
        vivaPitchText.textContent = `"${verdict.viva_elevator_pitch}"`;

        previewFilename.textContent = data.filename;

        // Render Media Preview thumbnail
        previewContainer.innerHTML = "";
        if (isVideo) {
            const vid = document.createElement("video");
            vid.src = mediaUrl || "";
            vid.controls = true;
            vid.muted = true;
            vid.autoplay = false;
            previewContainer.appendChild(vid);
        } else {
            const img = document.createElement("img");
            img.src = mediaUrl || (data.ela_metrics ? data.ela_metrics.ela_image_base64 : "");
            previewContainer.appendChild(img);
        }

        // Image vs Video visualizers
        if (isVideo) {
            imgVisualizer.classList.add("hidden");
            videoVisualizer.classList.remove("hidden");
            renderVideoVisuals(data.video_metrics);
        } else {
            videoVisualizer.classList.add("hidden");
            imgVisualizer.classList.remove("hidden");
            renderImageVisuals(data, mediaUrl);
        }

        // Render Explainability Cards
        renderForensicCards(verdict.forensic_cards);
    }

    function renderImageVisuals(data, origMediaUrl) {
        const ela = data.ela_metrics || {};
        const fft = data.fft_metrics || {};
        const noise = data.noise_metrics || {};

        const displayOrig = origMediaUrl || "";

        // Tab 1: ELA
        elaOrigView.src = displayOrig;
        elaDiffView.src = ela.ela_image_base64 || "";

        // Tab 2: FFT
        fftOrigView.src = displayOrig;
        fftSpecView.src = fft.fft_spectrum_base64 || "";

        // Tab 3: Noise
        noiseOrigView.src = displayOrig;
        noiseResView.src = noise.noise_map_base64 || "";
    }

    function renderVideoVisuals(vm) {
        if (!vm) return;

        vstatSsim.textContent = vm.avg_ssim !== undefined ? vm.avg_ssim.toFixed(3) : "N/A";
        vstatJitter.textContent = vm.ssim_jitter !== undefined ? `σ ${vm.ssim_jitter.toFixed(3)}` : "N/A";
        vstatFrames.textContent = vm.total_frames || "0";
        vstatDuration.textContent = `${vm.duration_seconds || 0}s duration (${vm.fps || 24} fps)`;

        // Keyframe Snapshots
        if (vm.highest_anomaly_frame) {
            kfHighestImg.src = vm.highest_anomaly_frame.image_base64;
            kfHighestTime.textContent = `Timestamp: ${vm.highest_anomaly_frame.timestamp}`;
            kfHighestScore.textContent = `AI Score: ${vm.highest_anomaly_frame.score.toFixed(1)}%`;
        }
        if (vm.lowest_anomaly_frame) {
            kfLowestImg.src = vm.lowest_anomaly_frame.image_base64;
            kfLowestTime.textContent = `Timestamp: ${vm.lowest_anomaly_frame.timestamp}`;
            kfLowestScore.textContent = `AI Score: ${vm.lowest_anomaly_frame.score.toFixed(1)}%`;
        }

        // Draw Canvas Timeline Graph
        drawTemporalChart(vm.timeline || []);
    }

    function drawTemporalChart(timeline) {
        if (!temporalCanvas || timeline.length === 0) return;
        const ctx = temporalCanvas.getContext("2d");
        const w = temporalCanvas.width;
        const h = temporalCanvas.height;

        ctx.clearRect(0, 0, w, h);

        // Draw background grid lines
        ctx.strokeStyle = "#1b2438";
        ctx.lineWidth = 1;
        for (let y = 30; y < h - 20; y += 35) {
            ctx.beginPath();
            ctx.moveTo(40, y);
            ctx.lineTo(w - 20, y);
            ctx.stroke();
        }

        const paddingLeft = 50;
        const paddingRight = 30;
        const paddingTop = 25;
        const paddingBottom = 30;
        const plotW = w - paddingLeft - paddingRight;
        const plotH = h - paddingTop - paddingBottom;

        // Draw axes labels
        ctx.fillStyle = "#64748b";
        ctx.font = "11px 'JetBrains Mono', monospace";
        ctx.fillText("SSIM 1.0", 5, paddingTop + 8);
        ctx.fillText("SSIM 0.5", 5, paddingTop + plotH / 2);
        ctx.fillText("SSIM 0.0", 5, paddingTop + plotH);

        // Plot points
        const count = timeline.length;
        if (count < 2) return;

        const points = timeline.map((pt, i) => {
            const x = paddingLeft + (i / (count - 1)) * plotW;
            const normSSIM = Math.max(0, Math.min(1, pt.ssim));
            const y = paddingTop + (1.0 - normSSIM) * plotH;
            return { x, y, pt };
        });

        // Draw smooth gradient curve
        ctx.beginPath();
        ctx.moveTo(points[0].x, points[0].y);
        for (let i = 1; i < points.length; i++) {
            ctx.lineTo(points[i].x, points[i].y);
        }
        ctx.strokeStyle = "#06b6d4";
        ctx.lineWidth = 2.5;
        ctx.stroke();

        // Fill area under curve
        ctx.lineTo(points[points.length - 1].x, paddingTop + plotH);
        ctx.lineTo(points[0].x, paddingTop + plotH);
        ctx.closePath();
        const grad = ctx.createLinearGradient(0, paddingTop, 0, paddingTop + plotH);
        grad.addColorStop(0, "rgba(6, 182, 212, 0.25)");
        grad.addColorStop(1, "rgba(6, 182, 212, 0.0)");
        ctx.fillStyle = grad;
        ctx.fill();

        // Draw point dots
        points.forEach((p) => {
            ctx.beginPath();
            ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
            // Red dot if SSIM dropped below 0.85 (flicker anomaly)
            if (p.pt.ssim < 0.85) {
                ctx.fillStyle = "#ef4444";
            } else {
                ctx.fillStyle = "#10b981";
            }
            ctx.fill();
            ctx.strokeStyle = "#0a0e17";
            ctx.lineWidth = 1.5;
            ctx.stroke();
        });
    }

    function renderForensicCards(cards) {
        forensicCardsList.innerHTML = "";

        cards.forEach((card) => {
            const cardEl = document.createElement("div");
            cardEl.className = "f-card";

            cardEl.innerHTML = `
                <div class="f-card-header">
                    <div class="f-card-title-wrap">
                        <span class="f-card-title">${card.title}</span>
                    </div>
                    <span class="f-card-badge badge ${card.badge_color}">${card.score}% (${card.status})</span>
                </div>
                <div class="f-card-body">
                    <div class="f-field">
                        <span class="f-label">Scope:</span>
                        <span class="f-val">${card.what_it_means}</span>
                    </div>
                    <div class="f-field">
                        <span class="f-label">Algorithm Finding:</span>
                        <span class="f-val" style="color: #e2e8f0; font-weight: 500;">${card.finding}</span>
                    </div>
                    <div class="f-field">
                        <span class="f-label">Why it Matters:</span>
                        <span class="f-val" style="color: #94a3b8; font-style: italic;">${card.why_it_matters}</span>
                    </div>
                </div>
            `;
            forensicCardsList.appendChild(cardEl);
        });
    }
});
