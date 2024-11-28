# # Main file for the project
# import streamlit as st
# import requests
# from streamlit_lottie import st_lottie

# html = """
#     <div style="background-color:#025246 ;padding:10px">
#     <h2 style="color:white;text-align:center;">AI Fitness Trainer</h2>
#     </div>"""
# st.markdown(html, unsafe_allow_html=True)

# col1, col2 = st.columns([10,8], gap="large")

# with col1:
#     st.write("## Trainer")


# frame_placeholder = st.empty()
# def load_lottieurl(url: str):
#     r = requests.get(url)
#     if r.status_code != 200:
#         return None
#     return r.json()

# lottie_hello = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_FYx0Ph.json")

# with col2:
#    st_lottie(lottie_hello,key="hello1", height=500, width=400)
import requests
import streamlit as st
from streamlit_lottie import st_lottie
from PIL import Image


# Find more emojis here: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="BizBoost Fitness Trainer", layout="wide")

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css(".\styles\styles.css")

# ---- LOAD ASSETS ----
lottie_coding = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_FYx0Ph.json")
music = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_ikk4jhps.json")
podcast = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_JjpNLdaKYX.json")


img_contact_form = Image.open("./images/home.jpg")
img_lottie_animation = Image.open("./images/home.jpg")

# ---- HEADER SECTION ----
with st.container():
    st.subheader("Advanced Fitness and Wellness Solutions")
    st.title("BizBoost Crew ~ Empowering a sustainable inclusive community")
    st.write(
        "Step into a fitter future. Welcome to your fitness revolution!"
    )
    #st.write("[Learn More >](https://pythonandvba.com)")

# ---- WHAT I DO ----
with st.container():
    
    left_column, right_column = st.columns(2)
    with left_column:
        st.header("")
        #st.write("##")
        st.write(
            """
            Welcome to BizBoost Crew, where technology meets fitness and wellness! We are a passionate team dedicated to empowering individuals and communities to achieve their health and fitness goals using cutting-edge AI technologies.
            
            Today, fitness is far more scientific. Body composition is analyzed with more precision, and health professionals use a variety of tools, from heart rate monitors to waist-to-hip ratio calculations, to help people understand their fitness levels. However, these tools still rely on basic measurements that may not fully capture one’s fitness progress.
            
            Let your fitness journey start here!
            
            [Click here to join us today!](https://www.facebook.com/profile.php?id=100068688709597)
            
            Embark on a transformative experience that will enhance your physical and mental well-being. Let's build strength, resilience, and a healthier future together!
           
            """
           
        )
    with right_column:
        st.image(
                    "images/logo.png",  # Replace with the URL or file path of your image
                    caption="Empowering Fitness with Technology",
                    width=450
                    # Add a smaller image to the right column
    
        )

with st.container():
    # Create a row of three columns
    col1, col2, col3 = st.columns(3)

    # Add images to each column
    with col1:
        st.image(
            "images/pic2.png",  # Replace with your image URL or file path
            width=300  # Adjust the width for uniform size
        )
    with col2:
        st.image(
            "images/pic1.png",  # Replace with your image URL or file path
            width=300  # Adjust the width for uniform size
        )
    with col3:
        st.image(
            "images/pic3.png",  # Replace with your image URL or file path
            width=300 
            
        )

st.title('AI Fitness Trainer: Squats Analysis')


recorded_file = 'output_sample.mp4'
sample_vid = st.empty()
sample_vid.video(recorded_file)

# ---- PROJECTS ----
with st.container():
    st.write("---")
    st.header("Get fit, Jam on, Repeat :headphones:")
    st.write("##")
    image_column, text_column = st.columns((1, 2))
    with image_column:
        #st.image(img_lottie_animation)
        st_lottie(music, height=300, key="music")
    with text_column:
        st.write("##")
        st.subheader("Workout music")
        st.write(
            """
            Power up your workout with the ultimate music fuel!
            """
        )
        st.markdown("[Have a Listen...](https://open.spotify.com/playlist/6N0Vl77EzPm13GIOlEkoJn?si=9207b7744d094bd3)")
with st.container():
    image_column, text_column = st.columns((1, 2))
    with image_column:
        #st.image(img_contact_form)
        st_lottie(podcast, height=300, key="podcast")
    with text_column:
        st.write("##")
        st.subheader("Podcast")
        st.write(
            """
            Take your workouts to the next level with our immersive podcast that pumps you up from start to finish!
            """
        )
        st.markdown("[Have a listen...](https://open.spotify.com/playlist/09Ig7KfohF5WmU9RhbDBjs?si=jyZ79y3wQgezrEDHim0NvQ)")

# ---- CONTACT ----
with st.container():
    st.write("---")
    st.header("Get In Touch With Me!")
    st.write("##")

    # Documention: https://formsubmit.co/ !!! CHANGE EMAIL ADDRESS !!!
    contact_form = """
    <form action="https://formsubmit.co/ayekhinkhinphone@gmail.com" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder="Your name" required>
        <input type="email" name="email" placeholder="Your email" required>
        <textarea name="message" placeholder="Your message here" required></textarea>
        <button type="submit">Send</button>
    </form>
    """
    left_column, right_column = st.columns(2)
    with left_column:
        st.markdown(contact_form, unsafe_allow_html=True)
    with right_column:
        st.empty()

# Footer with All Rights Reserved and Facebook Icon
st.write(
    """
      <div style='position: fixed; bottom: 0; left: 0; width: 100%; background-color: #d3d3d3; text-align: center; padding: 10px 0; font-size: 14px;'>
        <span>All rights reserved © 2024 BizBoost Crew</span>
        <a href="https://www.facebook.com/profile.php?id=100068688709597" target="_blank" style='margin: 0 15px;'>
            <img src="https://upload.wikimedia.org/wikipedia/commons/5/51/Facebook_f_logo_%282019%29.svg" alt="Facebook" style='width: 20px; height: 20px; vertical-align: middle;'>
        </a>
        <a href="https://line.me/en/" target="_blank" style='margin: 0 15px;'>
            <img src="https://scdn.line-apps.com/n/line_regulation/files/ver2/LINE_Icon.png" alt="LINE" style='width: 20px; height: 20px; vertical-align: middle;'>
        </a>
        <a href="mailto:example@example.com" target="_blank" style='margin: 0 15px;'>
            <img src="https://img.icons8.com/ios-filled/50/000000/email.png" alt="Email" style='width: 20px; height: 20px; vertical-align: middle;'>
        </a>
    </div>
    """,
    unsafe_allow_html=True
)