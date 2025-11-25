#No AI was used to generate this code
#All syntax was developed by HG on 11/25/25
import requests
import streamlit as st

API_URL = "https://behavioral-api-service-605345346838.us-east1.run.app/predict"


def build_payload(
    age,
    job,
    marital,
    education,
    default_flag,
    balance,
    housing_flag,
    loan_flag,
    contact,
    month,
    campaign,
    pdays,
    previous,
    poutcome,
):
    
#Building the full JSON payload expected by the FastAPI model. This matches the UCI-style schema your backend is using.
    

    payload = {
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default_flag,
        "balance": balance,
        "housing": housing_flag,
        "loan": loan_flag,
        "contact": contact,
        "month": month,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome,
    }

    return payload


def main():
    st.title("Behavioral Cashflow Reliability Tracker")
    st.write(
        "This app sends your inputs to a deployed ML model on Google Cloud Run "
        "to predict whether next month will be **surplus** or **deficit**."
    )

    st.subheader("Behavioral & Cashflow Inputs")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=45, step=1)
        job = st.selectbox(
            "Job",
            [
                "admin.",
                "blue-collar",
                "entrepreneur",
                "housemaid",
                "management",
                "retired",
                "self-employed",
                "services",
                "student",
                "technician",
                "unemployed",
                "unknown",
            ],
            index=4,
        )
        marital = st.selectbox(
            "Marital status", ["single", "married", "divorced"], index=1
        )
        education = st.selectbox(
            "Education",
            ["primary", "secondary", "tertiary", "unknown"],
            index=2,
        )
        default_flag = st.selectbox("In default on credit?", ["no", "yes"], index=0)
        balance = st.number_input(
            "Current account balance (EUR)",
            min_value=-100000,
            max_value=1000000,
            value=1000,
            step=100,
        )

    with col2:
        housing_flag = st.selectbox("Has housing loan?", ["no", "yes"], index=1)
        loan_flag = st.selectbox("Has personal loan?", ["no", "yes"], index=0)
        contact = st.selectbox("Contact type", ["cellular", "telephone", "unknown"], index=0)
        month = st.selectbox(
            "Current month",
            ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
            index=4,
        )
        campaign = st.number_input(
            "Number of contacts during this campaign",
            min_value=1,
            max_value=50,
            value=2,
            step=1,
        )
        pdays = st.number_input(
            "Days since last contact (or -1 if never)",
            min_value=-1,
            max_value=999,
            value=10,
            step=1,
        )
        previous = st.number_input(
            "Number of prior contacts before this campaign",
            min_value=0,
            max_value=100,
            value=1,
            step=1,
        )
        poutcome = st.selectbox(
            "Outcome of previous campaign",
            ["success", "failure", "other", "unknown"],
            index=0,
        )

    if st.button("Predict surplus vs deficit"):
        payload = build_payload(
            age=age,
            job=job,
            marital=marital,
            education=education,
            default_flag=default_flag,
            balance=balance,
            housing_flag=housing_flag,
            loan_flag=loan_flag,
            contact=contact,
            month=month,
            campaign=campaign,
            pdays=pdays,
            previous=previous,
            poutcome=poutcome,
        )

        with st.spinner("Calling model API on Cloud Run..."):
            try:
                response = requests.post(API_URL, json=payload, timeout=10)
                if response.status_code != 200:
                    st.error(f"API error: {response.status_code}")
                    st.code(response.text)
                else:
                    result = response.json()
                    surplus_pred = result.get("surplus_prediction")
                    surplus_prob = result.get("surplus_probability")

                    if surplus_pred == 1:
                        label = "Surplus next month"
                    else:
                        label = "Deficit next month"

                    st.success(f"Prediction: {label}")
                    if surplus_prob is not None:
                        st.write(f"Model probability of surplus: **{surplus_prob:.3f}**")

                    st.subheader("Raw API response")
                    st.json(result)

            except Exception as e:
                st.error(f"Error calling API: {e}")


if __name__ == "__main__":
    main()

