import streamlit as st
import cv2
import math
import numpy as np
import mediapipe as mp
import plotly.graph_objects as go


# Defining a class for finding angles between body landmarks
class AngleFinder:
    def __init__(self, landmarks, p1, p2, p3, draw_points, img):
        self.landmarks = landmarks
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.draw_points = draw_points
        self.img = img

    def angle(self):
        if self.landmarks:
            try:
                # Fetch the coordinates of the relevant landmarks
                x1, y1 = self.landmarks[self.p1].x * self.img.shape[1], self.landmarks[self.p1].y * self.img.shape[0]
                x2, y2 = self.landmarks[self.p2].x * self.img.shape[1], self.landmarks[self.p2].y * self.img.shape[0]
                x3, y3 = self.landmarks[self.p3].x * self.img.shape[1], self.landmarks[self.p3].y * self.img.shape[0]

                # Calculate the angle using trigonometry
                angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
                angle = int(np.interp(angle, [42, 143], [100, 0]))

                # Draw points and lines if needed
                if self.draw_points:
                    self._draw_points((int(x1), int(y1)), (int(x2), int(y2)), (int(x3), int(y3)))

                return angle
            except Exception as e:
                st.write(f"Error in calculating angle: {e}")
                return 0
        else:
            st.write("No landmarks detected.")
            return 0

    def _draw_points(self, p1, p2, p3):
        cv2.circle(self.img, p1, 10, (0, 255, 255), 5)
        cv2.circle(self.img, p1, 15, (0, 255, 0), 6)
        cv2.circle(self.img, p2, 10, (0, 255, 255), 5)
        cv2.circle(self.img, p2, 15, (0, 255, 0), 6)
        cv2.circle(self.img, p3, 10, (0, 255, 255), 5)
        cv2.circle(self.img, p3, 15, (0, 255, 0), 6)
        cv2.line(self.img, p1, p2, (0, 0, 255), 4)
        cv2.line(self.img, p2, p3, (0, 0, 255), 4)


# Initialize session states
def initialize_session():
    for i in range(1, 6):
        if f'counter{i}' not in st.session_state:
            st.session_state[f'counter{i}'] = 0
        if f'type{i}' not in st.session_state:
            st.session_state[f'type{i}'] = None


# Improved video capture function using MediaPipe with enhanced UI
def video_capture(exercise_num, points, weight, goal_calories):
    frame_placeholder = st.empty()
    counter, direction = 0, 0

    # Initialize MediaPipe Pose detection
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, img = cap.read()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (640, 480))

        # Process the image to detect pose landmarks
        results = pose.process(img_rgb)
        landmarks = results.pose_landmarks.landmark if results.pose_landmarks else None

        angle_finder = AngleFinder(landmarks, *points, draw_points=True, img=img)
        angle = angle_finder.angle()

        # Counting logic based on angle
        if angle >= 90 and direction == 0:
            counter += 0.5
            st.session_state[f'counter{exercise_num}'] = counter
            direction = 1
        if angle <= 70 and direction == 1:
            counter += 0.5
            st.session_state[f'counter{exercise_num}'] = counter
            direction = 0

        # Improved UI layout: counter box and text
        h, w, _ = img.shape

        # Semi-transparent rectangle for the counter
        overlay = img.copy()
        cv2.rectangle(overlay, (w - 180, 0), (w, 100), (0, 0, 0), -1)  # Black background for the counter
        alpha = 0.4  # Transparency factor
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        # Display counter and other text on the screen
        cv2.putText(img, "Reps:", (w - 170, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, str(int(counter)), (w - 80, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 3)

        # Display instructional text (if any) at the bottom
        cv2.putText(img, "Keep proper posture!", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Display the updated frame with the counter and text
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(img, "RGB")

        # Stop the stream if user presses Stop button
        if st.session_state[f'type{exercise_num}'] == 'Stop':
            cap.release()
            break

    pose.close()


# Function to display analytics after workout
def display_analytics(exercise_num, weight, goal_calories):
    st.write("The video capture has ended")
    st.write(f"You did {st.session_state[f'counter{exercise_num}']} reps")
    calories_burned = 3.8 * weight / st.session_state[f'counter{exercise_num}']
    st.write(f"You have burned {calories_burned:.2f} kcal of calories")

    if calories_burned < goal_calories:
        st.write("You have not achieved your goal. Try again")
    else:
        st.write("You have achieved your goal. Congratulations")

    # Plot graph
    fig = go.Figure(data=[go.Bar(x=['Exercise'], y=[calories_burned], name='Calories Burned')])
    fig.add_trace(go.Bar(x=['Exercise'], y=[goal_calories], name='Goal Calorie'))
    fig.update_layout(title='Calories Burned for Exercise', xaxis_title='Exercise', yaxis_title='Calories Burned')
    st.plotly_chart(fig)


# Main Streamlit app
def main():
    initialize_session()
    st.sidebar.title("Workout Tracker")
    app_mode = st.sidebar.selectbox("Choose the exercise", ["About", "Left Dumbbell", "Right Dumbbell", "Squats", "Pushups", "Shoulder press"])

    if app_mode == "About":
        st.markdown("## Welcome to the Training Arena")
        st.write("""
            Choose the workout you wish to do from the sidebar. 
            Ensure that you provide webcam access and work out in a well-lit, uncluttered space.
        """)
        st.image('./gif/ham.gif')

    elif app_mode == "Left Dumbbell":
        st.markdown("## Left Dumbbell")
        weight = st.slider('What is your weight?', 20, 130, 40)
        goal_calories = st.slider('Set a goal calorie to burn', 10, 200, 15)

        st.button('Start', on_click=lambda: st.session_state.update({'type1': 'Start'}))
        st.button('Stop', on_click=lambda: st.session_state.update({'type1': 'Stop'}))

        if st.session_state['type1'] == 'Start':
            video_capture(1, [11, 13, 15], weight, goal_calories)
        elif st.session_state['type1'] == 'Stop':
            display_analytics(1, weight, goal_calories)


if __name__ == "__main__":
    main()
