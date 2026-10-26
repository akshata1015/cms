"""
Run with: streamlit run streamlit_app.py
Dashboard: live camera feed with boxes + alerts panel + alert log table.
"""

import os

# Must be BEFORE cv2 / crime_detector are imported: disable the flaky MSMF camera backend on Windows.
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")

import time
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from crime_detector import CrimeDetector, open_camera, resize_keep_aspect
from analytics import hotspot_risk, hourly_summary

try:
    import winsound
except ImportError:
    winsound = None

st.set_page_config(page_title="AI Crime Detection System", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --ink: #e8eef2;
    --muted: #8c9aa3;
    --panel: #151d22;
    --panel-2: #1b262c;
    --line: #2b3941;
    --cyan: #5dd6d1;
    --amber: #f2b45b;
    --red: #f06c65;
}

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
[data-testid="stAppViewContainer"] { background: #0d1317; color: var(--ink); }
[data-testid="stHeader"] { background: rgba(13, 19, 23, .85); }
[data-testid="stSidebar"] { background: #11191e; border-right: 1px solid var(--line); }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { font-family: 'Space Grotesk', sans-serif; }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: 0; }
h1 { font-size: 2.1rem !important; margin-bottom: .15rem; }
h2 { font-size: 1.25rem !important; }
h3 { font-size: 1rem !important; }
[data-testid="stMetric"] { background: var(--panel); border: 1px solid var(--line); padding: 14px 16px; border-radius: 8px; }
[data-testid="stMetricLabel"] { color: var(--muted); }
[data-testid="stMetricValue"] { color: var(--ink); font-family: 'Space Grotesk', sans-serif; }
[data-testid="stTabs"] button { color: var(--muted); font-weight: 600; }
[data-testid="stTabs"] button[aria-selected="true"] { color: var(--cyan); }
[data-testid="stFileUploader"] section, [data-testid="stExpander"] { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; }
.console-kicker { color: var(--cyan); font-size: .72rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; }
.console-subtitle { color: var(--muted); margin: 0 0 1.2rem; }
.status-bar { display: flex; gap: 8px; flex-wrap: wrap; margin: 0 0 1rem; }
.status-chip { background: var(--panel); border: 1px solid var(--line); border-radius: 999px; color: var(--muted); font-size: .78rem; padding: 5px 10px; }
.status-chip b { color: var(--cyan); }
.alert-panel { background: #21191b; border: 1px solid #593337; border-radius: 8px; padding: 14px 16px; }
.empty-panel { background: var(--panel); border: 1px dashed var(--line); border-radius: 8px; color: var(--muted); padding: 28px; text-align: center; }
.result-banner { background: linear-gradient(110deg, #17282c, #182024); border: 1px solid #31565a; border-radius: 8px; padding: 16px 18px; margin: 12px 0 16px; }
.result-banner strong { color: var(--cyan); font-family: 'Space Grotesk', sans-serif; }
.result-banner span { color: var(--muted); font-size: .85rem; }
.image-frame { background: #0a0f12; border: 1px solid var(--line); border-radius: 8px; padding: 8px; }
.mini-label { color: var(--muted); font-size: .72rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; margin: 4px 0 8px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="console-kicker">Command Center / Computer Vision</div>', unsafe_allow_html=True)
st.title("Crime Detection Dashboard")
st.markdown('<p class="console-subtitle">Real-time situational awareness for camera feeds, alerts and evidence.</p>', unsafe_allow_html=True)
st.markdown('''<div class="status-bar">
<span class="status-chip"><b>●</b> AI engine ready</span>
<span class="status-chip">YOLOv8 object analysis</span>
<span class="status-chip">LBPH face watchlist</span>
<span class="status-chip">ANPR enabled</span>
</div>''', unsafe_allow_html=True)

with st.sidebar:
    st.header("Detection settings")
    yolo_confidence = st.slider("Object confidence", 0.10, 0.90, 0.45, 0.05)
    face_confidence = st.slider("Face match distance", 30, 120, 75, 5)
    fight_distance = st.slider("Fight proximity", 40, 250, 100, 10)
    fight_motion = st.slider("Fight movement", 5, 60, 15, 5)
    crowd_limit = st.slider("Crowd alert limit", 3, 50, 8)
    location = st.text_input("Camera location", "Local Camera")
    source_type = st.radio("Video source", ["Webcam", "RTSP / IP camera"])
    rtsp_url = st.text_input("RTSP URL", "", type="password") if source_type.startswith("RTSP") else ""
    camera_index = st.number_input("Webcam index", min_value=0, max_value=5, value=0, step=1)
    process_every = st.slider("Run detection every N frames", 1, 5, 2,
                              help="Higher = faster on CPU, but detection updates less often.")
    st.caption("Lower face distance is stricter and reduces false matches.")

if "detector" not in st.session_state:
    with st.spinner("Loading AI models (first time only, may take 10-30 seconds)..."):
        st.session_state.detector = CrimeDetector(
            yolo_confidence=yolo_confidence,
            face_confidence_thresh=face_confidence,
            fight_distance_thresh=fight_distance,
            fight_motion_thresh=fight_motion,
            crowd_limit=crowd_limit,
            location=location,
            source="rtsp" if source_type.startswith("RTSP") else "webcam",
        )
    detector = st.session_state.detector
    detector.current_stats = {"person_count": 0, "crowd_alert": False, "vehicle_count": 0, "weapon_count": 0}

detector = st.session_state.detector
detector.yolo_confidence = yolo_confidence
detector.face_confidence_thresh = face_confidence
detector.fight_distance_thresh = fight_distance
detector.fight_motion_thresh = fight_motion
detector.crowd_limit = crowd_limit
detector.location = location

tab_live, tab_image = st.tabs(["Live camera", "Test an image"])

with tab_live:
    col1, col2 = st.columns([2, 1])
    with col1:
        run = st.checkbox("Start camera / live detection")
        frame_placeholder = st.empty()

    with col2:
        st.subheader("Live alerts")
        alert_placeholder = st.empty()
        st.subheader("Alert log")
        log_placeholder = st.empty()

    if run:
        if source_type.startswith("RTSP") and rtsp_url:
            camera_source = rtsp_url
        else:
            camera_source = int(camera_index)

        with st.spinner("Opening camera..."):
            cap = open_camera(camera_source)

        if cap is None:
            st.error("Camera open nahi jhala. Dusre app (Meet/Teams/Zoom/Chrome) band kara, "
                     "camera index badla, kinva Windows Privacy settings madhe camera access ON kara.")
        else:
            writer = None
            last_sound_time = 0
            os.makedirs("evidence", exist_ok=True)
            evidence_path = os.path.join("evidence", f"session_{datetime.now():%Y%m%d_%H%M%S}.mp4")

            frame_no = 0
            fails = 0
            max_fails = 30
            last_annotated = None
            last_faces = []
            fight = False
            plate_alerts = []
            wanted_matches = []

            try:
                while True:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        fails += 1
                        if fails > max_fails:
                            # camera stream broke - release and try to reopen once
                            cap.release()
                            cap = open_camera(camera_source)
                            if cap is None:
                                st.error("Camera connection lost and could not be reopened.")
                                break
                            fails = 0
                        time.sleep(0.05)
                        continue
                    fails = 0
                    frame_no += 1

                    frame = resize_keep_aspect(frame, 640)

                    # Heavy AI work only every N frames (CPU friendly)
                    if frame_no % process_every == 0 or last_annotated is None:
                        last_annotated, fight, plate_alerts = detector.process_frame(frame)
                    if frame_no % (process_every * 2) == 0:
                        last_faces = detector.check_wanted_faces(frame)
                    wanted_matches = last_faces

                    display = last_annotated.copy()
                    for name, (left, top, right, bottom) in wanted_matches:
                        cv2.rectangle(display, (left, top), (right, bottom), (0, 0, 255), 2)
                        cv2.putText(display, f"WANTED: {name}", (left, top - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    if writer is None:
                        height, width = display.shape[:2]
                        writer = cv2.VideoWriter(
                            evidence_path, cv2.VideoWriter_fourcc(*"mp4v"), 20, (width, height))
                    writer.write(display)

                    frame_placeholder.image(cv2.cvtColor(display, cv2.COLOR_BGR2RGB), channels="RGB")

                    alerts_text = ""
                    if fight:
                        alerts_text += "**FIGHT DETECTED / भांडण आढळले**\n\n"
                        if winsound and time.time() - last_sound_time > 5:
                            winsound.Beep(1200, 250)
                            last_sound_time = time.time()
                    if detector.current_stats["crowd_alert"]:
                        alerts_text += f"**CROWD ALERT / गर्दी: {detector.current_stats['person_count']}**\n\n"
                    for name, _ in wanted_matches:
                        alerts_text += f"**WANTED PERSON: {name}**\n\n"
                    for plate_text, is_stolen in plate_alerts:
                        alerts_text += f"**{'STOLEN VEHICLE' if is_stolen else 'Plate read'}: {plate_text}**\n\n"
                    if alerts_text:
                        alert_placeholder.markdown(f'<div class="alert-panel">{alerts_text}</div>', unsafe_allow_html=True)
                    else:
                        alert_placeholder.markdown('<div class="empty-panel">No active alerts</div>', unsafe_allow_html=True)

                    # Refresh the log table only occasionally (reading CSV every frame is slow)
                    if frame_no % 20 == 0:
                        try:
                            log_placeholder.dataframe(pd.read_csv(detector.log_file).tail(15), width="stretch")
                        except Exception:
                            pass
                    time.sleep(0.01)
            finally:
                # Always runs - even when Streamlit stops the script (checkbox unticked / page rerun).
                # This is what releases the camera so it is not "locked" next time.
                if writer is not None:
                    writer.release()
                if cap is not None:
                    cap.release()

            st.caption(f"Evidence saved: {evidence_path}")
    else:
        st.info("Enable the checkbox to start the camera.")

with tab_image:
    st.markdown("### Image investigation")
    st.caption("Upload a frame or a close-up number plate image to run a one-shot analysis.")
    uploaded = st.file_uploader("Upload a CCTV image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if uploaded:
        with st.spinner("AI is analysing the uploaded frame..."):
            image = cv2.imdecode(np.frombuffer(uploaded.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                st.error("This image could not be decoded. Please upload a JPG or PNG file.")
                st.stop()
            max_dimension = 800
            image_scale = min(1.0, max_dimension / max(image.shape[:2]))
            if image_scale < 1.0:
                image = cv2.resize(image, None, fx=image_scale, fy=image_scale, interpolation=cv2.INTER_AREA)
            try:
                annotated, fight, plate_alerts = detector.process_frame(image)
                if not plate_alerts:
                    height, width = image.shape[:2]
                    plate_text = detector.read_plate(image, 0, 0, width, height, fast=True)
                    if plate_text:
                        plate_alerts = [(plate_text, plate_text in detector.stolen_plates)]
                # IMPORTANT: scan faces on the ORIGINAL image, not on `annotated`
                # (YOLO boxes/text drawn on top confuse the face detector).
                face_scan = detector.scan_faces(image)
                wanted_matches = [(f["name"], f["box"]) for f in face_scan if f["matched"]]
                for f in face_scan:
                    left, top, right, bottom = f["box"]
                    if f["matched"]:
                        box_color, box_text = (0, 0, 255), f"WANTED: {f['name']}"
                    else:
                        box_color, box_text = (0, 255, 255), f"{f['name']}? d={f['distance']:.0f}"
                    cv2.rectangle(annotated, (left, top), (right, bottom), box_color, 2)
                    cv2.putText(annotated, box_text, (left, max(15, top - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
            except Exception as error:
                st.error(f"Analysis failed: {error}")
                st.stop()

        alert_items = []
        if fight:
            alert_items.append("Fight detected / भांडण आढळले")
        if detector.current_stats["crowd_alert"]:
            alert_items.append(f"Crowd alert: {detector.current_stats['person_count']} people")
        alert_items.extend(f"Wanted person: {name}" for name, _ in wanted_matches)
        alert_items.extend(
            f"{'STOLEN VEHICLE' if stolen else 'Plate read'}: {plate}"
            for plate, stolen in plate_alerts
        )
        if alert_items:
            st.markdown(
                f'<div class="result-banner"><strong>Analysis complete</strong><br>'
                f'<span>{" &nbsp; | &nbsp; ".join(alert_items)}</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="result-banner"><strong>Analysis complete</strong><br>'
                '<span>No high-priority alert found in this image.</span></div>',
                unsafe_allow_html=True,
            )

        image_col, result_col = st.columns(2)
        with image_col:
            st.markdown('<div class="mini-label">Source frame</div>', unsafe_allow_html=True)
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), channels="RGB", width="stretch")
        with result_col:
            st.markdown('<div class="mini-label">AI analysis</div>', unsafe_allow_html=True)
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", width="stretch")

        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Fight", "Detected" if fight else "Not detected")
        metric2.metric("Wanted matches", len(wanted_matches))
        metric3.metric("Plate reads", len(plate_alerts))

        st.subheader("Face scan details")
        if not detector.trained:
            st.warning("Wanted faces train zale nahit. wanted_faces/ folder madhe photos aahet ka te check kara.")
        elif not face_scan:
            st.warning("Ya image madhe konta hi face detect jhala nahi. Face motha, samor (frontal), "
                       "ujed prakashat pahije; mask, goggles, tupi, side-view nako.")
        else:
            st.dataframe(pd.DataFrame([
                {"closest_wanted": f["name"], "distance": round(f["distance"], 1),
                 "threshold": detector.face_confidence_thresh, "matched": f["matched"]}
                for f in face_scan
            ]), hide_index=True, width="stretch")
            st.caption("Distance jitka kami titka match changla. Distance threshold peksha kami asel tar WANTED dakhavto. "
                       "Sidebar madhe 'Face match distance' vadhva (jast = soppa match, pan chukichya match jast).")
        if plate_alerts:
            st.subheader("Number plate results")
            plate_table = pd.DataFrame(
                [{"plate": plate, "status": "STOLEN" if stolen else "Read"} for plate, stolen in plate_alerts]
            )
            st.dataframe(plate_table, hide_index=True, width="stretch")

with st.expander("Crime records and hotspot report"):
    rows = detector.alert_store.database.recent(500)
    if rows:
        report = pd.DataFrame(rows, columns=["timestamp", "alert_type", "details", "location", "source"])
        st.dataframe(report, width="stretch")
        report_col1, report_col2 = st.columns(2)
        with report_col1:
            st.subheader("Hotspot locations")
            st.dataframe(pd.DataFrame(hotspot_risk(rows), columns=["location", "alerts", "risk_score"]), width="stretch")
        with report_col2:
            st.subheader("Alert hours")
            st.dataframe(pd.DataFrame(hourly_summary(rows), columns=["hour", "alerts"]), width="stretch")
        csv_data = report.to_csv(index=False).encode("utf-8")
        st.download_button("Download crime report", csv_data, "crime_report.csv", "text/csv")
    else:
        st.info("No crime records yet.")