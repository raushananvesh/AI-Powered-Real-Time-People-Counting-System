# import streamlit as st


# import cv2
# from ultralytics import YOLO

# # Load YOLO model
# model = YOLO("yolov8n.pt")

# # Start webcam
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
    
#     if not ret:
#         print("Failed to grab frame")
#         break

#     # Run YOLO detection
#     # results = model(frame)
#     results = model(frame, classes=[0], verbose=False)

#     # Draw results on frame
#     annotated_frame = results[0].plot()

#     # Show output
#     cv2.imshow("YOLO Webcam", annotated_frame)

#     # Press 'q' to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# person_count = len(results[0].boxes)
# print("People count:", person_count)        

# # Release resources
# cap.release()
# cv2.destroyAllWindows()






import streamlit as st
import cv2
from ultralytics import YOLO

# Cache model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# Initialize session state
if "run" not in st.session_state:
    st.session_state.run = False
if "cap" not in st.session_state:
    st.session_state.cap = None

st.title("AI-Powered Real-Time People Counting System")

col1, col2 = st.columns(2)

with col1:
    if st.button("▶ Start Camera"):
        st.session_state.run = True
        # Initialize camera only if it's not already running
        if st.session_state.cap is None:
            st.session_state.cap = cv2.VideoCapture(0)

with col2:
    if st.button("⏹ Stop Camera"):
        st.session_state.run = False
        # Release immediately on button click
        if st.session_state.cap is not None:
            st.session_state.cap.release()
            st.session_state.cap = None
        st.rerun()

frame_placeholder = st.empty()
count_placeholder = st.empty()

# The logic loop
if st.session_state.run and st.session_state.cap is not None:
    cap = st.session_state.cap
    
    while st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera feed lost.")
            break

        # Inference
        results = model(frame, classes=[0], verbose=False)
        person_count = len(results[0].boxes)
        
        annotated_frame = results[0].plot()
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Update UI
        frame_placeholder.image(annotated_frame, channels="RGB")
        count_placeholder.markdown(f"### People Count: {person_count}")

    # Safety release if the loop breaks for other reasons
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None
    frame_placeholder.empty()