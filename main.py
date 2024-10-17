import os
import streamlit as st
import google.generativeai as genai
from openai import OpenAI
from PIL import Image as PILImage

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets['OPENAI_API_KEY'])

# Initialize Gemini food_care_model
genai.configure(api_key=st.secrets['GOOGLE_API_KEY'])

baby_sleeping_model = genai.GenerativeModel("gemini-1.5-flash",
                              system_instruction="""
                                You are a baby care assistant. 
                                You need to analyze the baby's sleeping pattern based on the input information (sleeping position of the baby, age, and sleeping time).
                                Assess whether this sleeping pattern is good for the baby or not. If it is bad, list the disadvantages and suggest solutions to improve the sleeping pattern. 
                                Based on the sleeping time provided, assess whether it is appropriate for the baby's age, and if not, suggest the best sleeping time.
                              """
                             )

# OpenAI baby sleeping function
def baby_sleeping(age, sleeping_time, prompt):
    system_prompt = """
    You are a baby care assistant. 
    You need to analyze the baby's sleeping pattern based on the input information (sleeping position of the baby, age, and sleeping time).
    Assess whether this sleeping pattern is good for the baby or not. If it is bad, list the disadvantages and suggest solutions to improve the sleeping pattern. 
    Based on the sleeping time provided, assess whether it is appropriate for the baby's age, and if not, suggest the best sleeping time.
    """

    user_message = f"Baby Age: {age} years/months. Sleeping Time: {sleeping_time}. Sleeping Pattern: {prompt}. "

    response = client.chat.completions.create(
          model='gpt-4o-mini',
          messages=[
              {'role': 'system', 'content': system_prompt},
              {'role': 'user', 'content': user_message}
          ],
          temperature=1.0,
          max_tokens=2000
    )
    return response.choices[0].message.content

# Baby Sleeping Analyst Title
st.title("Baby Sleeping Analyst")

# Handle input based on user selection
uploaded_image = st.file_uploader("Choose a sleeping position image", type=["jpg", "jpeg", "png"])

# Display the uploaded image
if uploaded_image:
    img = PILImage.open(uploaded_image)
    img.thumbnail((800, 800))  # Resize image for display
    st.image(img, caption="Uploaded Baby Sleeping Position", use_column_width=True)

    # Generate description from image (make sure this part is compatible with the Gemini API)
    img_description = baby_sleeping_model.generate_content(["Describe the image", img])
    baby_pattern_description = img_description.text  # This should be the text description of the sleeping pattern

# Input for baby's age
baby_age = st.number_input("Baby's age (in months):", min_value=0, max_value=48, step=1)

# Input for sleeping time
baby_sleeping_time = st.number_input("Sleeping Time (in hours):", min_value=0, max_value=24, step=1)

# Submit button to run the function
if st.button("Check Sleep"):
    if (uploaded_image is not None) and (baby_age > 0):
        try:
            # Use the generated description if an image is uploaded
            result = baby_sleeping(baby_age, baby_sleeping_time, baby_pattern_description)

            # Display the result from the baby sleeping function
            st.write(result)
        except Exception as e:
            st.write("An error occurred:", str(e))
    else:
        st.write("Please provide an image of the baby and their age.")
