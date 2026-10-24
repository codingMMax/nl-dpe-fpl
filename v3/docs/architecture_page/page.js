(function () {
  "use strict";
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };

  /* ---------------- what each box of a module diagram is, and where it is in the Verilog ---------------- */
  var SUBS = {
    g_in: { t: "Token burst (E0)", x: "LOADX = ⌈8·min(d_m, R)/40⌉ words per token on V lanes of 40 bits; lane v carries rows vR … vR+R−1 of the token. A burst starts only in a cycle with x_rdy high, then never stops (burst on ready).", l: ["top/rtl/sim/att_head.v:48|LOADX", "ip/gemm/rtl/sim/gemm_top.v:14|interface"] },
    g_tiles: { t: "V row tiles", x: "for gv < V (and gh < H = 1): one dpe per tile in REGULAR mode, extent ROWS_USED × COLS_USED = min(d_m, R) × d. Tile v multiplies its R rows of X by its R rows of W; all tiles take the same burst in lock-step.", l: ["ip/gemm/rtl/sim/gemm_top.v:166|tiles", "ip/gemm/rtl/sim/gemm_top.v:89|V, H"] },
    g_ready: { t: "Readiness", x: "MSB_SA_Ready is the AND of all tiles' ready. att_ctrl holds x_rdy low until the copy's burst has ended, then raises it when the three GEMMs of the copy are ready and the credits allow.", l: ["ip/gemm/rtl/sim/gemm_top.v:198|readiness", "top/rtl/sim/att_ctrl.v:108|x_rdy"] },
    g_tree: { t: "Byte tree", x: "Each tile drains d bytes. They are sign-extended and folded pairwise in TREE_PIPE = log₂V registered stages; with V = 1 there is no stage.", l: ["ip/gemm/rtl/sim/gemm_top.v:254|tree", "ip/gemm/rtl/sim/gemm_top.v:100|TREE_PIPE"] },
    g_ser: { t: "Serializer and lanes", x: "The low byte of each column's sum goes to data_out, 5 bytes per cycle; out_valid marks each word. A row is ⌈d/5⌉ words (E1, E2 or E3).", l: ["ip/gemm/rtl/sim/gemm_top.v:321|serializer"] },
    g_drain: { t: "Drain markers", x: "The dpe has no per-word valid; reg_full and dpe_done are pushed through a marker pipeline as deep as the data path, so out_valid lines up with the words. (reg_full itself is a busy flag one cycle behind the words.)", l: ["ip/gemm/rtl/sim/gemm_top.v:206|drain phase", "ip/gemm/rtl/sim/gemm_top.v:370|pass end"] },

    d_aport: { t: "A port", qk: "E1: q8 rows, int8, row frame (one row of d per window), A_CH = P_proj channels, A_SLOTS = R_b (all of Q, never released); the staging credit is per NB-row group.", pv: "E5: log P windows, log bytes, window frame (NWEL/B_kv rows per window), A_CH = N_EXP channels, A_SLOTS = 2, released by each job.", l: ["top/rtl/sim/att_head.v:98|QK A params", "top/rtl/sim/att_head.v:123|PV A params"] },
    d_bport: { t: "B port", qk: "E2: k8 rows, int8, TRANS_B = 1 (the port carries rows of Kᵀ = columns of B), B_CH = P_proj, B_SLOTS = B (all of K).", pv: "E3: v8 rows, int8, TRANS_B = 0, B_CH = P_proj, B_SLOTS = B (all of V). Credit 4 rows per copy; the X bursts must start in token order.", l: ["top/rtl/sim/att_head.v:99|QK B params", "top/rtl/sim/att_head.v:124|PV B params"] },
    d_aconv: { t: "A converters", qk: "for gch < A_CH: dimm_conv = an input word register plus one identity crossbar in LOG mode, extent d × d. L = LOG(x) = trunc8(x − 1); the sign is recovered from the log byte. One row per pass; it keeps up because a GEMM row period is never shorter.", pv: "No converter on the PV A port: log P already is log bytes (A_DOM = 1), with sign + (A_SIGNED = 0). The port word goes straight to distribution.", l: ["ip/dimm/rtl/sim/dimm_tile_top.v:152|g_sa", "ip/dimm/rtl/sim/dimm_tile_top.v:618|dimm_conv"] },
    d_bconv: { t: "B converters", x: "for gch < B_CH: one LOG converter crossbar per channel, in both instances (K and V arrive as int8 from the GEMMs). One per GEMM copy.", l: ["ip/dimm/rtl/sim/dimm_tile_top.v:164|g_sb", "ip/dimm/rtl/sim/dimm_tile_top.v:618|dimm_conv"] },
    d_dist: { t: "Distribution", x: "Each pre-bit-plane stream word (the converter's drain, or the port word of a log operand) reaches every slice one cycle later (DIST_LAT = 1). Each slice keeps the words that hold its columns.", l: ["ip/dimm/rtl/sim/dimm_tile_top.v:139|§1 streams"] },
    d_aload: { t: "A loader control (shared)", qk: "Transposer of NB-row groups (NB = I/B_kv rows per farm window): a group is staged two deep, then each slice emits K_s column words, one per cycle. A freed group pulses a_cr (E1, CAP 2 groups).", pv: "Window stager: per channel two window entries; a window gives each slice E_w = (A_ROWS/NB)·K_s words, one per cycle. No staging credit: the window-frame port credits its slots, which jobs release.", l: ["ip/dimm/rtl/sim/dimm_tile_top.v:227|§2 A loader"] },
    d_bload: { t: "B loader control (shared)", qk: "Transposer of N-row groups: one group = one B matrix (B_kv rows of K); each slice emits K_s words. A freed matrix pulses b_cr (E2, CAP 2 matrices).", pv: "Row writer: two row registers per channel; rows are written in row order, one per cycle, into the slice that owns the key; b_cr per row (E3, CAP 4 per copy, X in token order).", l: ["ip/dimm/rtl/sim/dimm_tile_top.v:360|§3 B loader", "ip/dimm/rtl/sim/dimm_tile_top.v:423|row writer"] },
    d_job: { t: "Job logic", x: "A one-entry job register (ja_q, jb_q; burst on ready, j_rdy = empty). The decision is registered: both slots valid in every slice, every slice idle, and an output credit → go_q starts every slice. Since 2026-10-15 the slices load their slots from ja_q / jb_q while idle (Fmax fix 1), so no start-cycle mux sits on a memory address.", l: ["ip/dimm/rtl/sim/dimm_tile_top.v:484|§4 jobs", "ip/dimm/rtl/sim/dimm_tile_top.v:497|dec"] },
    d_astage: { t: "A staging (per slice)", qk: "dimm_xpose_stage: for gk < G = NB banks and gi < K_s columns, the slice's columns of each row, two buffers; row t goes to bank t mod G, so rows arriving together on several channels land in different banks (P_proj ≤ G).", pv: "dimm_wstage: per channel two entries × A_ROWS rows × K_s keys, in flip-flops; the channels take turns on the slice's write port.", l: ["ip/dimm/rtl/sim/dimm_tile_top.v:552|instance", "ip/dimm/rtl/sim/dimm_tile_top.v:658|dimm_xpose_stage", "ip/dimm/rtl/sim/dimm_tile_top.v:706|dimm_wstage"] },
    d_bstage: { t: "B staging (per slice)", qk: "dimm_xpose_stage with G = N = B_kv rows: the slice's columns of one K matrix, two buffers; then K_s words, word k = byte k of each staged row.", pv: "None: the shared row writer writes each V row straight into the B memory of the slice that owns its key.", l: ["ip/dimm/rtl/sim/dimm_tile_top.v:565|B staging"] },
    d_amem: { t: "A memories", x: "X_S/A_GRP banks per slice, blocked: crossbar f reads memory ⌊f/A_GRP⌋ at index f mod A_GRP. Each holds A_SLOTS × T_W × A_GRP words of NB bytes at {slot, wave, index}; 1 write, 1 read. A_GRP is the largest power of two ≤ min(⌈I/5⌉, X_S).", l: ["ip/dimm/rtl/sim/dimm_tile_top.v:929|g_a", "ip/dimm/rtl/sim/dimm_tile_top.v:928|a_ra"] },
    d_bmem: { t: "B memories", x: "K_s/B_GRP memories per slice, each B_SLOTS × B_GRP words of N bytes at {slot, index}; B matrix i fills slot i mod B_SLOTS. B_GRP = min(A_GRP, K_s).", l: ["ip/dimm/rtl/sim/dimm_tile_top.v:879|g_bm", "ip/dimm/rtl/sim/dimm_tile_top.v:885|ra"] },
    d_engine: { t: "Window engine copy", x: "dimm_tile_ctrl, one per slice, led by the slice's crossbar 0: issues T_W waves per job; take-now acceptance (a window starts in the cycle the leader is ready), feed overlapping the previous drain, a depth-2 drain queue. All slices start in the same cycle and stay in lock-step without signals between them. The DIMM's own VTR path #1 (7.58 ns) runs from the leader's MSB_SA_Ready through this logic.", l: ["ip/dimm/rtl/sim/dimm_tile_top.v:823|instance", "ip/dimm/rtl/sim/dimm_tile_top.v:1038|dimm_tile_ctrl"] },
    d_xbars: { t: "Farm crossbars", x: "for gf < XS: crossbar f works on column k = ⌊f/NG⌋ of group f mod NG for the whole tile. Per crossbar: the feed word computed one cycle ahead, u = trunc8(L_A + L_B) over I elements, dpe EXP (I × I, identity weights), a 4-entry A sign ring, and the registered signed drain ±e.", l: ["ip/dimm/rtl/sim/dimm_tile_top.v:945|g_x", "ip/dimm/rtl/sim/dimm_tile_top.v:980|dpe"] },
    d_ptree: { t: "Partial trees", x: "dimm_tree over the slice's K_s columns: the first log₂K_s registered adder levels of each group.", l: ["ip/dimm/rtl/sim/dimm_tile_top.v:1031|instance", "ip/dimm/rtl/sim/dimm_tile_top.v:1146|dimm_tree"] },
    d_ttree: { t: "Top tree", x: "The last log₂n_s levels across the slices, registered like the levels inside a slice, so the total depth is log₂K.", l: ["ip/dimm/rtl/sim/dimm_tile_top.v:585|§6"] },
    d_narrow: { t: "Output width", qk: "NARROW = 5: s8 = sat8(acc ≫ 5), the int8 scores (in place of 1/√d).", pv: "NARROW = 0: full width, 9 + log₂B_kv = 13 bits (o_part).", l: ["ip/dimm/rtl/sim/dimm_tile_top.v:598|g_out"] },
    d_cout: { t: "Output channels", qk: "E4: NG_QK = N_E/d channels to the online softmax (its S_CH input FIFOs); window w of a job on channel w mod NG; c_cr per job, C_CAP = 2 tiles.", pv: "E8: NG_PV = N_EP/B_kv channels to the accumulator (its P_CH FIFOs); C_CAP = 2 tiles.", l: ["ip/dimm/rtl/sim/dimm_tile_top.v:598|g_out", "top/rtl/sim/att_head.v:99|QK C_CAP", "top/rtl/sim/att_head.v:124|PV C_CAP"] },

    s_in: { t: "Input stage", x: "softmax_online_instage: one FIFO per E4 channel (S_CH = NG_QK), two tiles deep; the reader takes window w from FIFO w mod S_CH, one word per cycle, in window order; a 20-byte packer forms the block stream (S·B_kv bytes, 5 per word). s_cr pulses after a tile's last read.", l: ["ip/softmax/rtl/sim/softmax_online_top.v:1442|instance", "ip/softmax/rtl/sim/softmax_online_top.v:1184|module"] },
    s_gate: { t: "Block gate", x: "A block's stream starts when the run is armed, the store has a slot (block b − 2 has retired), a log P credit is free, and for b ≥ 1 an α credit is free. The gates sit in the input stage's reader.", l: ["ip/softmax/rtl/sim/softmax_online_top.v:1399|run arm and gates"] },
    s_store: { t: "Score store", x: "softmax_online_store: a 2-block ring, max(2, N_EXP) banks × 5 byte-lane RAMs, window-local layout; 1 write, 1 read.", l: ["ip/softmax/rtl/sim/softmax_online_top.v:1449|instance", "ip/softmax/rtl/sim/softmax_online_top.v:131|module"] },
    s_max: { t: "Row max", x: "softmax_online_loadmax: the block max per row while the block streams in, the running max m, the m_new and log α tables (ping-pong by block parity), and the E6 events (log α of blocks b ≥ 1).", l: ["ip/softmax/rtl/sim/softmax_online_top.v:1461|instance", "ip/softmax/rtl/sim/softmax_online_top.v:260|module"] },
    s_exp: { t: "EXP feeds and crossbars", x: "for gc < N_EXP: softmax_online_xfeed reads a window from the store, subtracts m_new, clamps to −128 and fills a 4-deep FIFO; its EXP crossbar (I × I) pops the FIFO when ready. The popped words are the log P stream on E5 lane gc, which is why E5 cannot stall and needs its credit.", l: ["ip/softmax/rtl/sim/softmax_online_top.v:1478|gen_exp", "ip/softmax/rtl/sim/softmax_online_top.v:436|xfeed", "ip/softmax/rtl/sim/softmax_online_top.v:530|crossbar"] },
    s_sum: { t: "Row sums", x: "softmax_online_row_acc: sp per row from the EXP drains, a 2-stage lane-sum tree, ping-pong by block.", l: ["ip/softmax/rtl/sim/softmax_online_top.v:1503|instance", "ip/softmax/rtl/sim/softmax_online_top.v:745|module"] },
    s_carry: { t: "Carry", x: "softmax_online_carry: b = 0: lq = min(sp ≫ LN, 127). b ≥ 1: l' = (carry-EXP(clamp8(log α + log l)) ≪ LN) + sp. Then carry-LOG gives log l. The carry crossbars use the extent S × S (one value per row); it retires blocks in order.", l: ["ip/softmax/rtl/sim/softmax_online_top.v:1550|instance", "ip/softmax/rtl/sim/softmax_online_top.v:956|module", "ip/softmax/rtl/sim/softmax_online_top.v:870|carry crossbar"] },
    s_ll: { t: "log l out (E7)", x: "The final log l of the run, read row by row through a registered read once ll_cr allows; done follows the last E7 word.", l: ["ip/softmax/rtl/sim/softmax_online_top.v:1326|top"] },

    a_fifo: { t: "o_part FIFOs", x: "for gf < P_CH: one att_fifo per E8 channel (P_CH = NG_PV), two tiles deep, words of 5 × 13 bits; window p of a tile sits in FIFO p mod P_CH. p_cr pulses after a tile's last read.", l: ["ip/softmax/rtl/sim/softmax_online_acc.v:187|g_f", "ip/lib/rtl/sim/att_fifo.v:14|att_fifo"] },
    a_ring: { t: "log α ring", x: "E6 fills a ring of 2 slots × S rows of 9-bit log α, in arrival order (not cleared at a run's end); la_cr pulses after a slot's last use.", l: ["ip/softmax/rtl/sim/softmax_online_acc.v:127|g_la"] },
    a_ll: { t: "log l table", x: "E7 fills S rows of 8-bit log l; ll_cr pulses after the normalization's last read.", l: ["ip/softmax/rtl/sim/softmax_online_acc.v:132|g_ll"] },
    a_log: { t: "LOG group", x: "N_OC identity crossbars in LOG mode (I × I), lock-step from crossbar 0. Job (b, w): feed o' = (b ? ±e ≪ SO : 0) + o_part, oq = min(|o'| ≫ SO, 127), so = [o' < 0]; drain lo → lo_mem. A job reads its N_OC windows from N_OC different FIFOs, so the RTL needs N_OC ≤ P_CH.", l: ["ip/softmax/rtl/sim/softmax_online_acc.v:172|engine", "ip/softmax/rtl/sim/softmax_online_acc.v:434|LOG crossbar", "ip/softmax/rtl/sim/softmax_online_acc.v:110|guard"] },
    a_exp: { t: "EXP group", x: "N_OC crossbars in EXP mode. Rescale job (b ≥ 1, w): t = clamp8(log α + lo), drain e → e_mem. Normalize job (w): tE = clamp8(lo − log l), drain O8 = sat8(±e) → e_mem. Every EXP job of a run also waits until the previous run's O8 has been read out of e_mem.", l: ["ip/softmax/rtl/sim/softmax_online_acc.v:177|engine", "ip/softmax/rtl/sim/softmax_online_acc.v:450|EXP crossbar"] },
    a_mem: { t: "Slot memories", x: "for gi < N_OC: per crossbar index lo_mem and e_mem (W·2^⌈log₂LW⌉ words of 40 bits) and so_mem. Window p = w·N_OC + i lives in crossbar i at wave w; W = ⌈P_oc/N_OC⌉ waves, P_oc = S·d/I windows per run, RPW = I/d rows per window.", l: ["ip/softmax/rtl/sim/softmax_online_acc.v:340|g_x", "ip/softmax/rtl/sim/softmax_online_acc.v:346|memories"] },
    a_ord: { t: "O reader", x: "Token t sits in window ⌊t/RPW⌋ (crossbar p mod N_OC, wave ⌊p/N_OC⌋) at offset (t mod RPW)·d. The reader reads ⌈d/5⌉ + 1 words, realigns and sends ⌈d/5⌉ words per row, rows back to back in token order (E9); run_done pulses one cycle after the run's last row.", l: ["ip/softmax/rtl/sim/softmax_online_acc.v:272|O reader"] },

    c_run: { t: "Run control", x: "start is taken once per reset; the head runs from t_s + 1 until done; busy falls one cycle after done (the start port's credit).", l: ["top/rtl/sim/att_ctrl.v:71|run control"] },
    c_cr: { t: "Credit counters", x: "qa_n, qb_n and pb_n count the pulses of qa_cr (QK A groups), qb_cr (QK B matrices) and pb_cr (PV B rows). A pulse at t counts for admission from t + 1. Several copies share one counter: a copy compares instead of decrementing.", l: ["top/rtl/sim/att_ctrl.v:80|counters"] },
    c_adm: { t: "Admission per copy", x: "for gc < P_PROJ: the copy's next token tok (c, c + P_proj, …) and its burst word counter wc. x_rdy[c] = running ∧ wc = 0 ∧ gq, gk, gv ready ∧ tok < n ∧ qa_n + 2 > ⌊tok/NB_A⌋ ∧ qb_n + 2 > ⌊tok/B_kv⌋ ∧ pb_n + 4·P_proj > tok. A burst starts when x_v rises and lasts LOADX words.", l: ["top/rtl/sim/att_ctrl.v:91|admission", "top/rtl/sim/att_ctrl.v:105|qa_ok", "top/rtl/sim/att_ctrl.v:106|qb_ok", "top/rtl/sim/att_ctrl.v:107|pb_ok", "top/rtl/sim/att_ctrl.v:108|x_rdy"] },
    c_jobs: { t: "Job lists", x: "Offered from the run start and taken on j_rdy: QK job j = (A slot ⌊j/B⌋, B slot j mod B, no release); PV job j = (A slot j mod 2 released, B slot j mod B); j = r·B + b, R_b·B jobs each.", l: ["top/rtl/sim/att_ctrl.v:113|job lists"] },
    c_runs: { t: "Softmax runs and done", x: "Run r < R_b starts in the first cycle with sm_busy low (sm_start lasts one cycle); done one cycle after the run_done of run R_b − 1.", l: ["top/rtl/sim/att_ctrl.v:142|runs"] }
  };

  /* ---------------- copy buttons ---------------- */
  function copyText(text, el) {
    function done() { el.classList.add("copied"); setTimeout(function () { el.classList.remove("copied"); }, 900); }
    try { navigator.clipboard.writeText(text).then(done, function () { selectFallback(el); }); } catch (e) { selectFallback(el); }
  }
  function selectFallback(el) {
    try { var r = document.createRange(); r.selectNodeContents(el); var s = window.getSelection(); s.removeAllRanges(); s.addRange(r); } catch (e) {}
  }
  function locButton(loc, note) {
    var b = document.createElement("button");
    b.type = "button"; b.className = "locbtn"; b.title = "Copy v3/" + loc; b.textContent = loc;
    if (note) { var i = document.createElement("i"); i.textContent = " " + note; b.appendChild(i); }
    b.addEventListener("click", function () { copyText("v3/" + loc, b); });
    return b;
  }
  $$("[data-trace]").forEach(function (el) {
    el.classList.add("trace");
    el.getAttribute("data-trace").split(";").forEach(function (item) { var p = item.split("|"); el.appendChild(locButton(p[0], p[1] || "")); });
  });
  $$("[data-loc]").forEach(function (el) { el.appendChild(locButton(el.getAttribute("data-loc"), "")); });

  /* ---------------- tabs ---------------- */
  var TABS = $$("nav.tabs a").map(function (a) { return a.getAttribute("data-tab"); });
  function showTab(name, keepScroll) {
    if (TABS.indexOf(name) < 0) name = "overview";
    $$(".view").forEach(function (v) { v.hidden = v.getAttribute("data-view") !== name; });
    $$("nav.tabs a").forEach(function (a) { if (a.getAttribute("data-tab") === name) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current"); });
    if (name === "doc") renderDoc();
    if (!keepScroll) { try { window.scrollTo(0, 0); } catch (e) {} }
  }
  function tabFromHash() { var h = (location.hash || "").replace("#", ""); return h || "overview"; }
  window.addEventListener("hashchange", function () { showTab(tabFromHash()); });
  function go(name) { if (location.hash !== "#" + name) location.hash = name; else showTab(name); }

  /* ---------------- zoom ---------------- */
  $$(".zoom").forEach(function (z) {
    var fig = document.getElementById("fig-" + z.getAttribute("data-zoom"));
    $$("button", z).forEach(function (b) {
      b.addEventListener("click", function () {
        var f = parseFloat(b.getAttribute("data-z"));
        $$("button", z).forEach(function (o) { o.setAttribute("aria-pressed", o === b ? "true" : "false"); });
        var svg = fig.querySelector("svg");
        svg.style.width = (100 * f) + "%";
        svg.style.minWidth = (960 * f) + "px";
      });
    });
  });

  /* ---------------- DIMM instance switch ---------------- */
  var variant = "qk";
  function setVariant(v) {
    variant = v;
    $$("#dimm-variant button").forEach(function (b) { b.setAttribute("aria-pressed", b.getAttribute("data-v") === v ? "true" : "false"); });
    $$("#fig-dimm [data-" + v + "]").forEach(function (el) { el.textContent = el.getAttribute("data-" + v); });
    var det = $("#det-dimm");
    if (det && det.getAttribute("data-sub")) showSub("dimm", det.getAttribute("data-sub"));
  }
  $$("#dimm-variant button").forEach(function (b) { b.addEventListener("click", function () { setVariant(b.getAttribute("data-v")); }); });

  /* ---------------- details panels ---------------- */
  function viewOf(el) { var v = el.closest(".view"); return v ? v.getAttribute("data-view") : "overview"; }
  function clearHl(view) { $$("#v-" + view + " .hl").forEach(function (e) { e.classList.remove("hl"); }); }
  function showSub(view, id) {
    var s = SUBS[id], det = $("#det-" + view);
    if (!s || !det) return;
    clearHl(view);
    $$('#v-' + view + ' .sub[data-sub="' + id + '"]').forEach(function (e) { e.classList.add("hl"); });
    det.setAttribute("data-sub", id);
    det.innerHTML = "";
    var k = document.createElement("div"); k.className = "k"; k.textContent = "Selected block"; det.appendChild(k);
    var h = document.createElement("h3"); h.textContent = s.t; det.appendChild(h);
    var p = document.createElement("p"); p.textContent = s.x || (variant === "pv" ? s.pv : s.qk); det.appendChild(p);
    if (!s.x && s.qk) { var n = document.createElement("p"); n.className = "note"; n.textContent = "Showing the " + (variant === "pv" ? "PV" : "QK") + " instance; switch above the diagram for the other."; det.appendChild(n); }
    var tr = document.createElement("div"); tr.className = "trace";
    s.l.forEach(function (it) { var q = it.split("|"); tr.appendChild(locButton(q[0], q[1] || "")); });
    det.appendChild(tr);
  }
  var PCARD = function (key) { return $('#v-params [data-param="' + key + '"]'); };
  function showParam(view, key) {
    var card = PCARD(key), det = $("#det-" + view);
    if (!card || !det) return;
    clearHl(view);
    $$('#v-' + view + ' .chip[data-param="' + key + '"]').forEach(function (e) { e.classList.add("hl"); });
    if (view === "overview") {
      (card.getAttribute("data-blocks") || "").split(/\s+/).forEach(function (b) {
        $$('#v-overview .blk[data-goto="' + (b === "pv" || b === "qk" ? "dimm" : (b === "sm" ? "softmax" : b)) + '"]').forEach(function (e) {
          var vv = e.getAttribute("data-variant");
          if (!vv || vv === b || (b !== "qk" && b !== "pv")) e.classList.add("hl");
        });
      });
    }
    det.removeAttribute("data-sub");
    det.innerHTML = "";
    var k = document.createElement("div"); k.className = "k"; k.textContent = "Selected parameter"; det.appendChild(k);
    var name = card.querySelector("h4") || card.querySelector("td");
    var h = document.createElement("h3"); h.innerHTML = name ? name.innerHTML : key; det.appendChild(h);
    if (card.tagName === "P") {
      var pp = document.createElement("p"); pp.innerHTML = card.innerHTML; det.appendChild(pp);
    } else if (card.tagName === "TR") {
      var c = card.querySelectorAll("td");
      var p0 = document.createElement("p"); p0.innerHTML = "<b>" + c[1].innerHTML + "</b>: " + c[2].innerHTML + " (example " + c[3].innerHTML + ")"; det.appendChild(p0);
    } else {
      $$(".pbody > p", card).forEach(function (p) { det.appendChild(p.cloneNode(true)); });
      var tbl = card.querySelector(".tablewrap"); if (tbl) det.appendChild(tbl.cloneNode(true));
    }
    var tr = card.querySelector(".trace");
    if (tr) { var t = document.createElement("div"); t.className = "trace"; $$("button", tr).forEach(function (b) { var loc = b.title.replace("Copy v3/", ""); var note = b.querySelector("i"); t.appendChild(locButton(loc, note ? note.textContent.trim() : "")); }); det.appendChild(t); }
    $$(".trace", det).forEach(function (tt) { $$("button", tt).forEach(function (b) { if (!b.onclick) { b.addEventListener("click", function () { copyText(b.title.replace("Copy ", ""), b); }); } }); });
    var a = document.createElement("a"); a.href = "#params"; a.textContent = "All parameters";
    a.addEventListener("click", function (e) { e.preventDefault(); go("params"); setTimeout(function () { card.scrollIntoView({ block: "center" }); card.classList.add("hl"); }, 30); });
    det.appendChild(a);
  }
  function onKey(fn) { return function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fn(); } }; }
  $$("svg .chip").forEach(function (c) {
    var key = c.getAttribute("data-param"), view = viewOf(c);
    var fn = function () { showParam(view, key); };
    c.addEventListener("click", function (e) { e.stopPropagation(); fn(); });
    c.addEventListener("keydown", onKey(fn));
  });
  $$("svg .sub").forEach(function (g) {
    var id = g.getAttribute("data-sub"), view = viewOf(g);
    var fn = function () { showSub(view, id); };
    g.addEventListener("click", fn); g.addEventListener("keydown", onKey(fn));
  });
  $$("svg .blk").forEach(function (g) {
    var fn = function () { var v = g.getAttribute("data-variant"); if (v) setVariant(v); go(g.getAttribute("data-goto")); };
    g.addEventListener("click", fn); g.addEventListener("keydown", onKey(fn));
  });
  /* "Parameters here" pin lists */
  var CLS = { P_PROJ: "searched", N_E: "searched", N_EP: "fixed", N_EXP: "fixed", N_OC: "fixed", BKV: "fixed", X_S: "fixed", BQ: "fixed", CREDITS: "credit", SHAPE: "derived" };
  var LABEL = { P_PROJ: "P_proj", N_E: "N_E", N_EP: "N_EP", N_EXP: "N_EXP", N_OC: "N_OC", BKV: "B_kv", X_S: "X_S", BQ: "B_q = S", CREDITS: "credit sizes",
                SHAPE: "shape, config", I: "I", S: "S", RB: "R_b", B: "B", V: "V", LOADX: "LOADX", NG: "NG", NS: "n_s, K_s", WK: "NB, W_K, T_W",
                AGRP: "A_GRP, B_GRP", PCH: "P_CH, P_oc", W: "W", LN: "LN = SO" };
  $$("[data-pins]").forEach(function (el) {
    var view = viewOf(el);
    el.getAttribute("data-pins").split(/\s+/).forEach(function (key) {
      var card = PCARD(key); if (!card) return;
      var b = document.createElement("button"); b.type = "button"; b.className = "pill " + (CLS[key] || "derived");
      b.textContent = LABEL[key] || key;
      b.addEventListener("click", function () { showParam(view, key); var det = $("#det-" + view); if (det) det.scrollIntoView({ block: "nearest" }); });
      el.appendChild(b);
    });
  });
  $$("#v-params .param .pname h4").forEach(function (h) {
    h.style.cursor = "default";
  });

  /* ---------------- explorer ---------------- */
  var DATA = [];
  try { DATA = JSON.parse($("#builds").textContent); } catch (e) { DATA = []; }
  var shapes = [], cfgs = [];
  DATA.forEach(function (r) { if (shapes.indexOf(r.shape) < 0) shapes.push(r.shape); var c = r.R + "×" + r.C; if (cfgs.indexOf(c) < 0) cfgs.push(c); });
  function fill(sel, items, val) { sel.innerHTML = ""; items.forEach(function (it) { var o = document.createElement("option"); o.textContent = it; o.value = it; sel.appendChild(o); }); if (val) sel.value = val; }
  var fmt = function (x) { return (typeof x === "number") ? x.toLocaleString("en-US") : String(x); };
  function find(shape, R, C, pp, ne, nep) {
    for (var i = 0; i < DATA.length; i++) { var r = DATA[i]; if (r.shape === shape && r.R === R && r.C === C && r.P_PROJ === pp && r.N_E === ne && r.N_EP === nep) return r; }
    return null;
  }
  function kv(pairs) { return '<div class="kv">' + pairs.map(function (p) { return "<span>" + p[0] + "</span><span>" + fmt(p[1]) + "</span>"; }).join("") + "</div>"; }
  function render() {
    if (!DATA.length) return;
    var shape = $("#x-shape").value, cfg = $("#x-cfg").value.split("×"), R = +cfg[0], C = +cfg[1];
    var any = DATA.filter(function (r) { return r.shape === shape; })[0], d = any.d;
    var pp = +$("#x-pp").value, ne = +$("#x-ne").value * d, nep = +$("#x-nep").value * d;
    var r = find(shape, R, C, pp, ne, nep), ref = find(shape, R, C, 1, 2 * d, 2 * d);
    var v = $("#x-verdict"), bars = $("#x-bars"), cards = $("#x-cards");
    var head = "n " + any.n + ", d<sub>m</sub> " + any.dm + ", d " + d + " at " + R + "×" + C + ", P<sub>proj</sub> " + pp + ", N<sub>E</sub> " + ne + ", N<sub>EP</sub> " + nep;
    if (!r || r.error) {
      v.innerHTML = '<span><b class="bad">Outside the RTL domain</b></span><span>' + head + '</span><span class="note">' + (r ? r.error : "no data") + '</span>';
      bars.innerHTML = ""; cards.innerHTML = ""; return;
    }
    var dc = ref && ref.cycles ? (100 * (r.cycles - ref.cycles) / ref.cycles) : 0, dx = ref ? r.n_xbar - ref.n_xbar : 0;
    var sgn = function (x, u) { return (x > 0 ? "+" : "") + (u === "%" ? x.toFixed(1) + " %" : x); };
    v.innerHTML = '<span><b>' + fmt(r.cycles) + '</b> cycles (model)</span><span><b>' + r.n_xbar + '</b> crossbars</span>' +
      '<span class="note">' + head + '</span><span class="note">vs the reference mapping: ' + sgn(dc, "%") + ' cycles, ' + sgn(dx) + ' crossbars</span>';
    var parts = [["GEMMs (3·P·V)", r.gemm_xbar], ["QK DIMM (N_E + 2P)", r.qk.xbar], ["online softmax (N_EXP + 2)", r.sm.xbar], ["PV DIMM (N_EP + P)", r.pv.xbar], ["accumulator (2·N_OC)", r.acc.xbar]];
    var mx = 0; DATA.forEach(function (x) { if (x.qk && x.qk.xbar > mx) mx = x.qk.xbar; });
    bars.innerHTML = parts.map(function (p) { return '<div class="bar"><span>' + p[0] + '</span><div class="track"><div class="fill" style="width:' + (100 * p[1] / mx).toFixed(1) + '%"></div></div><span class="num">' + p[1] + '</span></div>'; }).join("");
    cards.innerHTML =
      '<div class="mini"><h4>Head</h4>' + kv([["I", r.I], ["S (rows per run)", r.S], ["R_b runs", r.RB], ["B key blocks", r.B], ["V row tiles", r.V], ["LOADX words", r.LOADX], ["NG_QK", r.NG_QK], ["NG_PV = P_CH", r.NG_PV], ["N_OC", r.N_OC]]) + '</div>' +
      '<div class="mini"><h4>QK DIMM</h4>' + kv([["NB rows/window", r.qk.NB], ["W_K windows/column", r.qk.W_K], ["NG groups", r.qk.NG], ["T_W waves", r.qk.T_W], ["n_s slices", r.qk.NS], ["K_s columns/slice", r.qk.KS], ["A_GRP · B_GRP", r.qk.A_GRP + " · " + r.qk.B_GRP], ["start latency", r.qk.START_LAT], ["memory bits", r.qk.mem_bits]]) + '</div>' +
      '<div class="mini"><h4>PV DIMM</h4>' + kv([["NB rows/window", r.pv.NB], ["W_K windows/column", r.pv.W_K], ["NG groups", r.pv.NG], ["T_W waves", r.pv.T_W], ["n_s slices", r.pv.NS], ["K_s keys/slice", r.pv.KS], ["A_GRP · B_GRP", r.pv.A_GRP + " · " + r.pv.B_GRP], ["start latency", r.pv.START_LAT], ["memory bits", r.pv.mem_bits]]) + '</div>' +
      '<div class="mini"><h4>Softmax and accumulator</h4>' + kv([["windows per block", r.sm.NWB], ["store banks", r.sm.NB], ["input FIFO words", r.sm.DF], ["softmax memory bits", r.sm.mem_bits], ["P_oc windows/run", r.acc.P_OC], ["W waves", r.acc.W], ["rows per window", r.acc.RPW], ["accumulator memory bits", r.acc.mem_bits]]) + '</div>';
  }
  if (DATA.length) {
    fill($("#x-shape"), shapes, "BERT-Tiny"); fill($("#x-cfg"), cfgs, "128×128");
    ["#x-shape", "#x-cfg", "#x-pp", "#x-ne", "#x-nep"].forEach(function (s) { $(s).addEventListener("change", render); });
    render();
  }

  /* ---------------- the document ---------------- */
  var docDone = false;
  var ANNO = {
    "2.1 Symbols": ["I", "S", "RB", "B", "V", "P_PROJ", "N_E", "N_EP", "N_EXP", "N_OC"],
    "3.3 Edges": ["CREDITS", "NG"],
    "4.3 GEMM (gemm_top)": ["P_PROJ", "V", "LOADX"],
    "4.4 Tiled DIMM (dimm_tile_top)": ["N_E", "N_EP", "X_S", "NG", "NS", "AGRP", "BKV"],
    "4.5 Online softmax (softmax_online_top)": ["N_EXP", "BKV", "BQ"],
    "4.6 Online softmax accumulator (softmax_online_acc)": ["N_OC", "PCH", "W"],
    "4.7 Tiling controller (att_ctrl) and top (att_head)": ["P_PROJ", "S", "RB", "B"],
    "5.2 Credits and fixed sizes": ["CREDITS", "N_OC"]
  };
  var GOTO = { "4.3 GEMM (gemm_top)": "gemm", "4.4 Tiled DIMM (dimm_tile_top)": "dimm", "4.5 Online softmax (softmax_online_top)": "softmax",
               "4.6 Online softmax accumulator (softmax_online_acc)": "acc", "4.7 Tiling controller (att_ctrl) and top (att_head)": "ctrl" };
  function renderDoc() {
    if (docDone) return;
    docDone = true;
    var src = $("#md-src").textContent;
    $("#doc-fallback").textContent = src;
    if (!(window.marked && typeof window.marked.parse === "function")) return;
    var math = [];
    var s2 = src.replace(/\$\$([\s\S]+?)\$\$/g, function (m, t) { math.push([true, t]); return "@@MATH" + (math.length - 1) + "@@"; });
    s2 = s2.replace(/\$([^\$\n]+?)\$/g, function (m, t) { math.push([false, t]); return "@@MATH" + (math.length - 1) + "@@"; });
    var esc = function (t) { return t.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); };
    var html = window.marked.parse(s2);
    html = html.replace(/@@MATH(\d+)@@/g, function (m, i) { var x = math[+i]; return x[0] ? "\\[" + esc(x[1]) + "\\]" : "\\(" + esc(x[1]) + "\\)"; });
    var body = $("#doc-body"); body.innerHTML = html;
    var toc = $("#doc-toc");
    $$("h2, h3", body).forEach(function (h, i) {
      var t = h.textContent.trim(); h.id = "d-" + i;
      if (h.tagName === "H2") { var a = document.createElement("a"); a.href = "#doc"; a.textContent = t; a.addEventListener("click", function (e) { e.preventDefault(); h.scrollIntoView(); }); toc.appendChild(a); }
      var key = Object.keys(ANNO).filter(function (k) { return t === k; })[0];
      if (key) {
        var bar = document.createElement("div"); bar.className = "anno";
        if (GOTO[key]) { var g = document.createElement("a"); g.href = "#" + GOTO[key]; g.textContent = "Open the module view"; bar.appendChild(g); bar.appendChild(document.createTextNode(" · ")); }
        bar.appendChild(document.createTextNode("Defines:"));
        ANNO[key].forEach(function (pk) {
          var b = document.createElement("button"); b.type = "button"; b.className = "pill " + (CLS[pk] || "derived");
          b.textContent = LABEL[pk] || pk;
          b.addEventListener("click", function () { go("overview"); setTimeout(function () { showParam("overview", pk); }, 20); });
          bar.appendChild(b);
        });
        h.parentNode.insertBefore(bar, h.nextSibling);
      }
    });
    var typeset = function () { if (window.MathJax && window.MathJax.typesetPromise) window.MathJax.startup.promise.then(function () { return window.MathJax.typesetPromise([body]); }).catch(function () {}); };
    if (document.readyState === "complete") typeset(); else window.addEventListener("load", typeset);
  }

  showTab(tabFromHash(), true);
})();
