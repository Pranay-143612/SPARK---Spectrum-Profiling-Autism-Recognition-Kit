# import os
# import time
# import pickle
# import cv2
# import pandas as pd
# import streamlit as st
# import tensorflow as tf
# import numpy as np
# import altair as alt
# from streamlit_option_menu import option_menu

# # Set page configuration
# st.set_page_config(page_title="SPARK",
#                    layout="wide",
#                    page_icon="👨🏻‍⚕️")

# # Inject custom styles including orange color for h1
# # Inject custom CSS
# st.markdown("""
#     <style>
#         .custom-title {
#             text-align: center;
#             color: #f63366;
#             font-size: 2.5em;
#             font-weight: bold;
#         }
#         h4 {
#             text-align: center;
#         }
#         .result-card { 
#             background-color: #e8f5e9; 
#             padding: 20px; 
#             border-radius: 10px; 
#         }
#         .autistic { 
#             color: #d32f2f; 
#         }
#         .not-autistic { 
#             color: #2e7d32; 
#         }
#     </style>
# """, unsafe_allow_html=True)

# # ---- Title Section ----
# st.markdown('<div class="custom-title">🧠 SPARK - Spectrum Profiling & Autism Recognition Kit</div>', unsafe_allow_html=True)
# st.markdown('<h4>Combining Questionnaire & Computer Vision for Smarter Diagnosis</h4>', unsafe_allow_html=True)
# st.markdown("---")




# # Define working directory
# working_dir = os.path.dirname(os.path.abspath(__file__))

# # Load the autism prediction model
# try:
#     autism_model = pickle.load(open(os.path.join(working_dir, 'Questionnaire_Analysis/autism_model.sav'), 'rb'))
# except Exception as e:
#     st.error(f"Error loading model: {e}")
#     st.stop()

# # Load the Computer Vision model
# try:
#     computer_vision_model = tf.keras.models.load_model(os.path.join(working_dir, 'Computer_Vision/saved_models/resnet50_autism_classifier.h5'))
# except Exception as e:
#     st.error(f"Error loading computer vision model: {e}")
#     st.stop()

# # Sidebar menu
# with st.sidebar:
#     selected = option_menu(
#         menu_title="SPARK",
#         options=['Questionnaire Analysis', 'Computer Vision', 'Result'],
#         icons=['list-task', 'eye'],  
#         menu_icon="hospital",
#         default_index=0,  # default selected menu item
#     )

# # Function to preprocess and predict using the computer vision model
# def predict_computer_vision(frame):
#     img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (224, 224))  # Assuming the model expects 224x224 input size
#     img = img / 255.0  # Normalize the image
#     img = np.expand_dims(img, axis=0)
#     prediction = computer_vision_model.predict(img)
#     confidence = prediction[0][0]
#     return confidence

# if 'qr' not in st.session_state:
#     st.session_state.qr = 0
# if 'cr' not in st.session_state:
#     st.session_state.cr = 0
# if 'patient_name' not in st.session_state:
#     st.session_state.patient_name = ""

# # Questionnaire Analysis section

# if selected == 'Questionnaire Analysis':
#     st.title('Questionnaire Analysis')

#     name = st.text_input("Name of the Patient")
#     if name:
#         st.session_state.patient_name = name

#     c1, c2 = st.columns(2)
#     yesorno = {"Yes": 1, "No": 0}

#     # Collect responses — initialized as None
#     responses = {}

#     with c1:
#         sex = st.radio("Gender", ["Male", "Female"], index=None)
#         responses["sex"] = {"Male": 1, "Female": 0}.get(sex)

#     with c2:
#         ethnicity = st.selectbox("Ethnicity", ["-- Select --"] + list({
#             "Hispanic": 0, "Latino": 1, "Native Indian": 2, "Others": 3, "Pacific": 4,
#             "White European": 5, "Asian": 6, "Black": 7, "Middle Eastern": 8, "Mixed": 9, "South Asian": 10
#         }.keys()))
#         ethnicity_mapping = {
#             "Hispanic": 0, "Latino": 1, "Native Indian": 2, "Others": 3, "Pacific": 4,
#             "White European": 5, "Asian": 6, "Black": 7, "Middle Eastern": 8, "Mixed": 9, "South Asian": 10
#         }
#         responses["ethnicity"] = ethnicity_mapping.get(ethnicity)

#     with c1:
#         responses["jaundice"] = yesorno.get(st.radio("Presence of Jaundice", ["Yes", "No"], index=None))
#     with c2:
#         responses["family_mem_with_ASD"] = yesorno.get(st.radio("Family member with ASD", ["Yes", "No"], index=None))

#     questions = [
#         "Does your child look at you when you call his/her name?",
#         "How easy is it for you to get eye contact with your child?",
#         "Does your child point to indicate that he/she wants something?",
#         "Does your child point to share interest with you?",
#         "Does your child pretend?",
#         "Does your child follow where you’re looking?",
#         "If someone in the family is visibly upset, does your child try to comfort them?",
#         "Were your child's first words clear and understandable?",
#         "Does your child use simple gestures?",
#         "Does your child stare at nothing with no apparent purpose?"
#     ]

#     for i, q in enumerate(questions):
#         col = c1 if i % 2 == 0 else c2
#         with col:
#             responses[f"a{i+1}"] = yesorno.get(st.radio(q, ["Yes", "No"], index=None, key=f"q{i+1}"))

#     if st.button('Autism Test Result'):
#         # Check if any response is None
#         if None in responses.values():
#             st.warning("⚠️ Please answer all questions and fill all required details before submitting.")
#         else:
#             try:
#                 inputs = [
#                     responses[f"a{i+1}"] for i in range(10)
#                 ] + [
#                     responses["sex"],
#                     responses["ethnicity"],
#                     responses["jaundice"],
#                     responses["family_mem_with_ASD"]
#                 ]

#                 autism_prediction = autism_model.predict([inputs])
#                 autism_confidence = autism_model.predict_proba([inputs])
#                 result = 'Autistic' if autism_prediction[0] == 0 else 'Not Autistic'
#                 confidence = autism_confidence[0][autism_prediction[0]] * 100

#                 st.session_state.qr = autism_confidence[0][autism_prediction[0]]

#                 st.subheader(f'{result}')
#                 st.metric(label="Model Confidence", value=f"{confidence:.2f}%")

#                 if name:
#                     st.subheader(f'Results for {name}')
#                 else:
#                     st.subheader('Results')

#                 data = {
#                     'Questions': questions,
#                     'Responses': ["Yes" if responses[f"a{i+1}"] == 1 else "No" for i in range(10)]
#                 }
#                 st.table(pd.DataFrame(data))

#             except Exception as e:
#                 st.error(f"Error during prediction: {e}")

# # Computer Vision
# # if selected == 'Computer Vision':
# #     st.title('Computer Vision')
# #     st.write("Instructions:")
# #     st.write("- Read the paragraph after starting the test.")
# #     st.write("- Click the 'Start Test' button to begin.")
    
# #     # Define a directory to save images
# #     img_dir = os.path.join(working_dir, 'captured_images')
# #     os.makedirs(img_dir, exist_ok=True)

# #     if st.button('Start Test'):
# #         st.markdown("""
# #         ## Read the paragraph
# #         Once, amidst the verdant pastures of Vrindavan, where the sweet fragrance of blooming flowers mingled
# #         with the melodious songs of birds, lived Lord Krishna, the epitome of divine love and wisdom. His 
# #         enchanting flute melodies echoed through the lush groves, captivating the hearts of both mortals and 
# #         celestial beings alike. The young cowherd, with his playful demeanor and profound teachings, drew 
# #         devotees from far and wide, each seeking solace in his divine presence. Legends spoke of his childhood
# #         antics, his endearing bond with cows, and his valorous deeds against the forces of darkness. 
# #         """)
        
# #         # Capture images for 10 seconds
# #         cap = cv2.VideoCapture(0)
# #         start_time = time.time()
# #         img_count = 0
# #         results = []

# #         while img_count < 10:
# #             ret, frame = cap.read()
# #             if not ret:
# #                 st.error("Failed to capture image")
# #                 break

# #             img_count += 1
# #             img_path = os.path.join(img_dir, f'image_{img_count}.jpg')
# #             cv2.imwrite(img_path, frame)

# #             # Validate the image using the classifier
# #             confidence = predict_computer_vision(frame)
# #             results.append(confidence)

# #             time.sleep(1)

# #         cap.release()
# #         st.success(f'Captured {img_count} images.')

# #         average_prediction = np.mean(results)
# #         st.session_state.cr = average_prediction
# #         threshold = 0.5
# #         final_result = 'Autistic' if average_prediction < threshold else 'Not Autistic'
        
# #         st.markdown(f"## Computer Vision Test Completed")
# #         st.subheader(f"Prediction : **{final_result}**")
        
# #         if final_result == 'Autistic':
# #             temp1 = 100 - average_prediction * 100
# #             st.metric(label="Model Confidence", value=f"{temp1:.2f}%")
# #         else:
# #             st.metric(label="Model Confidence", value=f"{average_prediction * 100:.2f}%")

# #         # Displaying images captured
# #         st.markdown("## Captured Images")
# #         st.write("Here are the images captured during the test:")

# #         image_files = sorted(os.listdir(img_dir))

# #         num_images = len(image_files)
# #         num_columns = 5
# #         num_rows = (num_images // num_columns) + (1 if num_images % num_columns != 0 else 0)

# #         for i in range(num_rows):
# #             columns = st.columns(num_columns)
# #             for j in range(num_columns):
# #                 index = i * num_columns + j
# #                 if index < num_images:
# #                     image_path = os.path.join(img_dir, image_files[index])
# #                     columns[j].image(image_path, use_column_width=True, caption=f"Image {index + 1}")

# #         # Display confidence levels
# #         st.markdown("## Computer Vision Confidence Levels")
# #         c1, c2, c3 = st.columns(3)
# #         with c1:
# #             st.write("Image Index")
# #             for i in range(1, img_count + 1):
# #                 st.write(i)
# #         with c2:
# #             st.write("Autistic (%)")
# #             for conf in results:
# #                 temp = 100 - conf * 100
# #                 st.write(f"{temp:.2f}%")
# #         with c3:
# #             st.write("Non-Autistic (%)")
# #             for conf in results:
# #                 st.write(f"{conf * 100:.2f}%")

# #         # Plot the results
# #         st.markdown("## Computer Vision Test Results")

# #         # Plotting using Altair
# #         df_results = pd.DataFrame({
# #             'Image Index': range(1, img_count + 1),
# #             'Prediction': results
# #         })
# #         line_chart = alt.Chart(df_results).mark_line(point=True).encode(
# #             x='Image Index',
# #             y='Prediction'
# #         ).properties(
# #             title='Computer Vision Predictions'
# #         )
# #         threshold_line = alt.Chart(pd.DataFrame({'y': [threshold]})).mark_rule(color='red').encode(y='y')
# #         st.altair_chart(line_chart + threshold_line, use_container_width=True)

# #         st.write("Thank you for participating in the computer vision test.")

# if selected == 'Computer Vision':
#     st.title('Computer Vision')
#     st.write("Instructions:")
#     st.write("- The webcam will be shown live for 20 seconds.")
#     st.write("- During that time, 10 images will be captured automatically (every 2 seconds).")
    
#     img_dir = os.path.join(working_dir, 'captured_images')
#     os.makedirs(img_dir, exist_ok=True)

#     if st.button('Start Test'):
#         st.info("📷 Webcam started... Please face the camera.")

#         cap = cv2.VideoCapture(0)
#         if not cap.isOpened():
#             st.error("Failed to access webcam.")
#         #capturing
#         else:
#             start_time = time.time()
#             last_capture = 0
#             img_count = 0
#             results = []
#             current_image_files = []

#             stframe = st.empty()

#             while time.time() - start_time < 20:
#                 ret, frame = cap.read()
#                 if not ret:
#                     st.error("Failed to read from webcam.")
#                     break

#                 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 stframe.image(frame_rgb, channels="RGB", caption="Live Feed", use_container_width=True)

#                 if time.time() - last_capture >= 2 and img_count < 10:
#                     img_count += 1
#                     last_capture = time.time()

#                     img_path = os.path.join(img_dir, f'image_{img_count}.jpg')
#                     cv2.imwrite(img_path, frame)
#                     current_image_files.append(img_path)

#                     confidence = predict_computer_vision(frame)
#                     results.append(confidence)

#             cap.release()
#             stframe.empty()
#             st.success(f'✅ Captured {img_count} images.')

#             average_prediction = np.mean(results)
#             st.session_state.cr = average_prediction
#             threshold = 0.5
#             final_result = 'Autistic' if average_prediction < threshold else 'Not Autistic'

#             st.markdown("## 🧠 Computer Vision Test Completed")
#             st.subheader(f"Prediction: **{final_result}**")

#             confidence_display = (1 - average_prediction) * 100 if final_result == 'Autistic' else average_prediction * 100
#             st.metric(label="Model Confidence", value=f"{confidence_display:.2f}%")

#             st.markdown("## 📸 Captured Images")
#             image_files = sorted(os.listdir(img_dir))
#             num_columns = 5
#             rows = (len(current_image_files) + num_columns - 1)// num_columns

#             for i in range(rows):
#                 cols = st.columns(num_columns)
#                 for j in range(num_columns):
#                     idx = i * num_columns + j
#                     if idx < len(image_files):
#                         cols[j].image(current_image_files[idx], use_container_width=True, caption=f"Image {idx + 1}")

#             st.markdown("## 🔍 Confidence Levels")
#             c1, c2, c3 = st.columns(3)
#             with c1:
#                 st.write("Image Index")
#                 for i in range(1, img_count + 1):
#                     st.write(i)
#             with c2:
#                 st.write("Autistic (%)")
#                 for conf in results:
#                     st.write(f"{(1 - conf) * 100:.2f}%")
#             with c3:
#                 st.write("Not Autistic (%)")
#                 for conf in results:
#                     st.write(f"{conf * 100:.2f}%")

#             st.markdown("## 📈 Prediction Plot")
#             df_results = pd.DataFrame({
#                 'Image Index': range(1, img_count + 1),
#                 'Prediction': results
#             })

#             line_chart = alt.Chart(df_results).mark_line(point=True).encode(
#                 x='Image Index',
#                 y='Prediction'
#             ).properties(title='Computer Vision Predictions')

#             threshold_line = alt.Chart(pd.DataFrame({'y': [threshold]})).mark_rule(color='red').encode(y='y')
#             st.altair_chart(line_chart + threshold_line, use_container_width=True)

#             st.write("✅ Thank you for participating in the computer vision test.")


# # Result section
# if selected == 'Result':
#     if st.session_state.patient_name:
#         st.title(f'Results of {st.session_state.patient_name}')
#     else:
#         st.title('Results')
#     try:
#         # Ensure the session states have been set
#         if 'qr' not in st.session_state and 'cr' not in st.session_state:
#             st.error("Questionnaire result or Computer Vision result not available.")
#         else:
#             # Log-odds conversion
#             qr_log_odds = np.log(st.session_state.qr / (1 - st.session_state.qr))
#             cr_log_odds = np.log(st.session_state.cr / (1 - st.session_state.cr))

#             # Combine the log odds
#             combined_log_odds = (qr_log_odds + cr_log_odds) / 2

#             # Convert back to probability
#             combined_probability = 1 / (1 + np.exp(-combined_log_odds))

#             result = 'Autistic' if combined_probability < 0.5 else 'Not Autistic'
#             confidence = combined_probability * 100 if combined_probability >= 0.5 else (1 - combined_probability) * 100

#             st.subheader(f'{result}')
#             st.metric(label="Confidence", value=f"{confidence:.2f}%")

#     except Exception as e:
#         st.error(f"Error during final prediction: {e}")

# spark_app.py

import os
import time
import pickle
import cv2
import pandas as pd
import streamlit as st
import tensorflow as tf
import numpy as np
import altair as alt
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title="SPARK", layout="wide", page_icon="👨🏻‍⚕️")

# Custom styles
st.markdown("""
    <style>
        .custom-title {
            text-align: center;
            color: #f63366;
            font-size: 2.5em;
            font-weight: bold;
        }
        h4 {
            text-align: center;
        }
        .result-card {
            background-color: #e8f5e9;
            padding: 20px;
            border-radius: 10px;
        }
        .autistic {
            color: #d32f2f;
        }
        .not-autistic {
            color: #2e7d32;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="custom-title">🧠 SPARK - Spectrum Profiling & Autism Recognition Kit</div>', unsafe_allow_html=True)
st.markdown('<h4>Combining Questionnaire & Computer Vision for Smarter Diagnosis</h4>', unsafe_allow_html=True)
st.markdown("---")

working_dir = os.path.dirname(os.path.abspath(__file__))

# Load models
try:
    autism_model = pickle.load(open(os.path.join(working_dir, 'Questionnaire_Analysis/autism_model.sav'), 'rb'))
except Exception as e:
    st.error(f"Error loading questionnaire model: {e}")
    st.stop()

try:
    computer_vision_model = tf.keras.models.load_model(os.path.join(working_dir, 'Computer_Vision/saved_models/resnet50_autism_classifier.h5'))
except Exception as e:
    st.error(f"Error loading vision model: {e}")
    st.stop()

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        menu_title="SPARK",
        options=['Questionnaire Analysis', 'Computer Vision', 'Result'],
        icons=['list-task', 'camera-video', 'check-circle'],
        menu_icon="hospital",
        default_index=0,
    )

# Session states
for key in ['qr', 'cr', 'patient_name']:
    if key not in st.session_state:
        st.session_state[key] = 0 if key != 'patient_name' else ""

# Prediction function
def predict_computer_vision(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = computer_vision_model.predict(img)
    return prediction[0][0]

# ----------------------------- Questionnaire Analysis -----------------------------
if selected == 'Questionnaire Analysis':
    st.title('📝 Questionnaire Analysis')
    name = st.text_input("Name of the Patient")
    if name:
        st.session_state.patient_name = name

    c1, c2 = st.columns(2)
    yesorno = {"Yes": 1, "No": 0}
    responses = {}

    with c1:
        sex = st.radio("Gender", ["Male", "Female"], index=None)
        responses["sex"] = {"Male": 1, "Female": 0}.get(sex)

    with c2:
        ethnicity_list = ["Hispanic", "Latino", "Native Indian", "Others", "Pacific",
                          "White European", "Asian", "Black", "Middle Eastern", "Mixed", "South Asian"]
        ethnicity = st.selectbox("Ethnicity", ["-- Select --"] + ethnicity_list)
        ethnicity_mapping = {eth: idx for idx, eth in enumerate(ethnicity_list)}
        responses["ethnicity"] = ethnicity_mapping.get(ethnicity)

    with c1:
        responses["jaundice"] = yesorno.get(st.radio("Presence of Jaundice", ["Yes", "No"], index=None))
    with c2:
        responses["family_mem_with_ASD"] = yesorno.get(st.radio("Family member with ASD", ["Yes", "No"], index=None))

    questions = [
        "Does your child look at you when you call his/her name?",
        "How easy is it for you to get eye contact with your child?",
        "Does your child point to indicate that he/she wants something?",
        "Does your child point to share interest with you?",
        "Does your child pretend?",
        "Does your child follow where you’re looking?",
        "If someone in the family is visibly upset, does your child try to comfort them?",
        "Were your child's first words clear and understandable?",
        "Does your child use simple gestures?",
        "Does your child stare at nothing with no apparent purpose?"
    ]

    for i, q in enumerate(questions):
        col = c1 if i % 2 == 0 else c2
        with col:
            responses[f"a{i+1}"] = yesorno.get(st.radio(q, ["Yes", "No"], index=None, key=f"q{i+1}"))

    if st.button('Get Questionnaire Result'):
        if None in responses.values():
            st.warning("⚠️ Please answer all questions and fill all required details before submitting.")
        else:
            try:
                inputs = [responses[f"a{i+1}"] for i in range(10)] + [
                    responses["sex"], responses["ethnicity"],
                    responses["jaundice"], responses["family_mem_with_ASD"]
                ]
                prediction = autism_model.predict([inputs])
                confidence = autism_model.predict_proba([inputs])[0][prediction[0]]
                st.session_state.qr = confidence
                result = "Autistic" if prediction[0] == 0 else "Not Autistic"
                st.subheader(f'{result}')
                st.metric("Model Confidence", f"{confidence * 100:.2f}%")
            except Exception as e:
                st.error(f"Error during prediction: {e}")

# ----------------------------- Computer Vision -----------------------------

if selected == 'Computer Vision':
    st.title('Computer Vision')
    st.write("Instructions:")
    st.write("- The webcam will be shown live for 20 seconds.")
    st.write("- During that time, 10 images will be captured automatically (every 2 seconds).")
    
    img_dir = os.path.join(working_dir, 'captured_images')
    os.makedirs(img_dir, exist_ok=True)

    if st.button('Start Test'):
        st.info("📷 Webcam started... Please face the camera.")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Failed to access webcam.")
        #capturing
        else:
            start_time = time.time()
            last_capture = 0
            img_count = 0
            results = []
            current_image_files = []

            stframe = st.empty()

            while time.time() - start_time < 20:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read from webcam.")
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame_rgb, channels="RGB", caption="Live Feed", use_container_width=True)

                if time.time() - last_capture >= 2 and img_count < 10:
                    img_count += 1
                    last_capture = time.time()

                    img_path = os.path.join(img_dir, f'image_{img_count}.jpg')
                    cv2.imwrite(img_path, frame)
                    current_image_files.append(img_path)

                    confidence = predict_computer_vision(frame)
                    results.append(confidence)

            cap.release()
            stframe.empty()
            st.success(f'✅ Captured {img_count} images.')

            average_prediction = np.mean(results)
            st.session_state.cr = average_prediction
            threshold = 0.5
            final_result = 'Autistic' if average_prediction < threshold else 'Not Autistic'

            st.markdown("## 🧠 Computer Vision Test Completed")
            st.subheader(f"Prediction: **{final_result}**")

            confidence_display = (1 - average_prediction) * 100 if final_result == 'Autistic' else average_prediction * 100
            st.metric(label="Model Confidence", value=f"{confidence_display:.2f}%")

            st.markdown("## 📸 Captured Images")
            image_files = sorted(os.listdir(img_dir))
            num_columns = 5
            rows = (len(current_image_files) + num_columns - 1)// num_columns

            for i in range(rows):
                cols = st.columns(num_columns)
                for j in range(num_columns):
                    idx = i * num_columns + j
                    if idx < len(image_files):
                        cols[j].image(current_image_files[idx], use_container_width=True, caption=f"Image {idx + 1}")

            st.markdown("## 🔍 Confidence Levels")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.write("Image Index")
                for i in range(1, img_count + 1):
                    st.write(i)
            with c2:
                st.write("Autistic (%)")
                for conf in results:
                    st.write(f"{(1 - conf) * 100:.2f}%")
            with c3:
                st.write("Not Autistic (%)")
                for conf in results:
                    st.write(f"{conf * 100:.2f}%")

            st.markdown("## 📈 Prediction Plot")
            df_results = pd.DataFrame({
                'Image Index': range(1, img_count + 1),
                'Prediction': results
            })

            line_chart = alt.Chart(df_results).mark_line(point=True).encode(
                x='Image Index',
                y='Prediction'
            ).properties(title='Computer Vision Predictions')

            threshold_line = alt.Chart(pd.DataFrame({'y': [threshold]})).mark_rule(color='red').encode(y='y')
            st.altair_chart(line_chart + threshold_line, use_container_width=True)

            st.write("✅ Thank you for participating in the computer vision test.")



# elif selected == 'Computer Vision':
#     st.title("📸 Computer Vision Analysis")
#     st.write("Instructions:")
#     st.write("- The webcam will be shown live for 20 seconds.")
#     st.write("- During that time, 10 images will be captured automatically (every 2 seconds).")

#     img_dir = os.path.join(working_dir, 'captured_images')
#     os.makedirs(img_dir, exist_ok=True)

#     if st.button("Start Test"):
#         st.info("📷 Webcam started... Please face the camera.")
#         cap = cv2.VideoCapture(0)

#         if not cap.isOpened():
#             st.error("Webcam not detected.")
#         else:
#             start_time = time.time()
#             last_capture = 0
#             img_count = 0
#             results = []
#             stframe = st.empty()

#             while time.time() - start_time < 20:
#                 ret, frame = cap.read()
#                 if not ret:
#                     st.error("Webcam read error.")
#                     break
#                 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 stframe.image(frame_rgb, channels="RGB", caption="Live Feed", use_container_width=True)

#                 if time.time() - last_capture >= 2 and img_count < 10:
#                     img_count += 1
#                     last_capture = time.time()
#                     img_path = os.path.join(img_dir, f'image_{img_count}.jpg')
#                     cv2.imwrite(img_path, frame)
#                     conf = predict_computer_vision(frame)
#                     results.append(conf)

#             cap.release()
#             avg_conf = np.mean(results)
#             st.session_state.cr = avg_conf
#             label = "Not Autistic" if avg_conf > 0.5 else "Autistic"
#             st.success(f"Prediction: **{label}**")
#             st.metric("Model Confidence", f"{avg_conf * 100:.2f}%")

#             # Chart
#             df = pd.DataFrame({
#                 "Image": list(range(1, len(results)+1)),
#                 "Non-Autistic Confidence": results
#             })
#             chart = alt.Chart(df).mark_line(point=True).encode(
#                 x="Image",
#                 y="Non-Autistic Confidence"
#             ).properties(title="Confidence Over Captures")
#             st.altair_chart(chart, use_container_width=True)

# ----------------------------- Final Combined Result -----------------------------
elif selected == 'Result':
    st.title("✅ Combined Result Summary")
    name = st.session_state.patient_name or "the patient"

    if st.session_state.qr == 0 and st.session_state.cr == 0:
        st.warning("Please complete both the Questionnaire and Computer Vision tests.")
    else:
        st.subheader(f"🧑‍⚕️ Final Diagnosis for {name}")
        qr = st.session_state.qr
        cr = st.session_state.cr

        # Weighted average (50% from each model)
        final_score = (qr + cr) / 2
        if final_score >= 0.5:
            final_result = "Not Autistic"
            color_class = "not-autistic"
        else:
            final_result = "Autistic"
            color_class = "autistic"

        st.markdown(f"""
        <div class="result-card">
            <h2 class="{color_class}">Final Result: {final_result}</h2>
            <p><strong>Questionnaire Confidence:</strong> {qr * 100:.2f}%</p>
            <p><strong>Computer Vision Confidence:</strong> {cr * 100:.2f}%</p>
            <hr>
            <h4>🧠 Combined Confidence: <span style='color:#3366cc'>{final_score * 100:.2f}%</span></h4>
        </div>
        """, unsafe_allow_html=True)
