import os
import random
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Makes dataset generation repeatable
random.seed(42)

SAMPLES_PER_CLASS = 300


# ============================================================
# DATASET
# ============================================================

texts = []
labels = []


# ============================================================
# COMMON INDIAN TERMS
# ============================================================

banks = [
    "SBI",
    "State Bank",
    "HDFC Bank",
    "ICICI Bank",
    "Axis Bank",
    "Kotak Bank",
    "PNB",
    "Bank of Baroda",
    "Canara Bank",
    "IDFC First Bank"
]

payment_apps = [
    "Paytm",
    "PhonePe",
    "Google Pay",
    "GPay",
    "BHIM UPI"
]

urgency = [
    "immediately",
    "right now",
    "today",
    "urgently",
    "within 2 hours",
    "within 24 hours",
    "as soon as possible",
    "before midnight",
    "without delay",
    "otherwise your account will be blocked"
]

hinglish_urgency = [
    "abhi",
    "jaldi",
    "turant",
    "aaj hi",
    "warna account band ho jayega",
    "warna UPI band ho jayega",
    "der mat karo",
    "abhi update karo",
    "jaldi se karo"
]


# ============================================================
# PHISHING
# ============================================================

phishing_templates = [

    "Your {bank} account has been suspended. Verify immediately.",
    "{bank} security alert: suspicious login detected.",
    "Your {bank} account has unusual activity. Login now.",
    "{bank} alert: confirm your identity {urgency}.",
    "Your account will be blocked. Verify your details now.",
    "Suspicious login detected from a new device.",
    "Your banking account has been compromised.",
    "Immediate verification required to restore account access.",
    "Click the link to secure your account.",
    "Your password needs to be reset immediately.",
    "Your account access has been restricted.",
    "Confirm your account details to avoid suspension.",
    "Security verification required {urgency}.",
    "Unusual activity detected on your bank account.",
    "Login now to prevent account closure.",
    "Your debit card will be blocked today.",
    "Your banking profile requires urgent verification.",
    "Verify your account before services are disabled.",
    "Security alert! Confirm your identity now.",
    "Your account has been temporarily locked.",
    "A suspicious transaction was detected. Verify now.",
    "Your online banking access will expire today.",
    "Please confirm your banking information.",
    "Click here to restore your account.",
    "Final warning: verify your account.",
    "Account verification pending. Take action now.",
    "Your account security needs immediate attention.",
    "Your bank account requires re-verification.",
    "Login immediately to secure your account.",
    "Verify your personal details to continue banking."
]


# ============================================================
# OTP SCAM
# ============================================================

otp_templates = [

    "Share your OTP to complete the transaction.",
    "Send OTP immediately to verify the payment.",
    "Bank executive asking for OTP.",
    "OTP required to unblock your account.",
    "Please tell me the OTP you received.",
    "Your OTP is required for verification.",
    "OTP nahi diya toh account band ho jayega.",
    "Apna OTP bhejo transaction complete karne ke liye.",
    "Sir OTP confirm kar do.",
    "OTP batao warna payment cancel ho jayega.",
    "Aapka OTP chahiye verification ke liye.",
    "OTP urgently send karo.",
    "Bank verification ke liye OTP share karo.",
    "UPI OTP bhejo jaldi.",
    "OTP confirm nahi kiya toh service band ho jayegi.",
    "Apna OTP bata do abhi.",
    "OTP forward karo verification ke liye.",
    "Transaction complete karne ke liye OTP chahiye.",
    "Account unlock karne ke liye OTP bhejo.",
    "OTP share karo warna account block ho jayega.",
    "Sir verification ke liye OTP send kijiye.",
    "Aapko jo OTP mila hai woh bataiye.",
    "OTP nahi bataya toh payment fail ho jayega.",
    "OTP required urgently.",
    "Send the verification OTP now.",
    "Please share the OTP for account verification.",
    "OTP needed to approve this transaction.",
    "OTP confirm karo jaldi.",
    "OTP bhejo warna service deactivate ho jayegi.",
    "Your OTP is required to complete this request."
]


# ============================================================
# KYC SCAM
# ============================================================

kyc_templates = [

    "{bank} KYC is incomplete. Update immediately.",
    "Your KYC has expired.",
    "KYC update required to continue banking services.",
    "Update your KYC before your account is blocked.",
    "Your KYC verification is pending.",
    "KYC nahi kiya toh account band ho jayega.",
    "Aapka KYC expire ho gaya hai.",
    "KYC update karo warna UPI band ho jayega.",
    "KYC verification ke liye link open karo.",
    "Your KYC will expire today.",
    "Complete KYC immediately to avoid suspension.",
    "KYC update required urgently.",
    "Bank account KYC pending.",
    "KYC incomplete hai, abhi update karein.",
    "Final warning: update KYC today.",
    "Aapka KYC pending hai turant update karo.",
    "KYC verify nahi hua toh banking service band.",
    "KYC update karna compulsory hai.",
    "KYC details verify karo abhi.",
    "KYC nahi hua toh account freeze ho jayega.",
    "Your account KYC needs immediate verification.",
    "Update KYC before midnight.",
    "KYC verification failed. Update details now.",
    "KYC expired please click the link.",
    "Bank KYC pending action required.",
    "KYC update karo warna account suspend hoga.",
    "Your KYC documents need verification.",
    "KYC complete karne ke liye details submit karo.",
    "Urgent KYC verification required.",
    "Last warning: your KYC is incomplete."
]


# ============================================================
# REFUND SCAM
# ============================================================

refund_templates = [

    "Your refund is pending. Verify your bank details.",
    "Refund failed. Update your account information.",
    "Your refund is ready. Click to claim.",
    "You are eligible for a refund.",
    "Refund processing requires bank verification.",
    "Send your bank details to receive the refund.",
    "Refund ke liye bank details bhejo.",
    "Aapka refund pending hai.",
    "Refund lene ke liye link open karo.",
    "Payment refund failed. Verify account.",
    "Your cashback refund is waiting.",
    "Claim your pending refund now.",
    "Refund amount is waiting for verification.",
    "UPI refund pending. Confirm details.",
    "Your refund will expire today.",
    "Refund lene ke liye account details submit karo.",
    "Aapka refund approve ho gaya hai claim karo.",
    "Refund process complete karne ke liye details do.",
    "Your refund cannot be processed without verification.",
    "Refund amount pending. Update bank information.",
    "Click here to receive your refund.",
    "Your payment refund requires confirmation.",
    "Refund request failed. Verify your details.",
    "Cashback refund available claim now.",
    "Refund pending for your recent transaction.",
    "Bank details required for refund.",
    "Refund processing stopped. Confirm account.",
    "Aapka paisa refund hona hai details bhejo.",
    "Refund ke liye verification zaroori hai.",
    "Claim your pending money immediately."
]


# ============================================================
# FAKE JOB
# ============================================================

job_templates = [

    "Work from home job. Earn 5000 daily.",
    "Earn 50000 per month from home.",
    "No experience required. Immediate hiring.",
    "Online typing job available.",
    "Part time job. Daily payment guaranteed.",
    "Telegram job opportunity. Join now.",
    "Earn money by completing simple tasks.",
    "Work from home and earn instantly.",
    "Limited vacancies. Apply immediately.",
    "You have been selected for an online job.",
    "No interview required. Start today.",
    "Data entry job with high salary.",
    "Earn 3000 to 10000 daily.",
    "Job available for students.",
    "Pay a small registration fee to get the job.",
    "Ghar baithe daily 5000 kamao.",
    "Part time job karo aur daily income pao.",
    "Work from home job no experience needed.",
    "Telegram par job available hai.",
    "Online task complete karo aur paisa kamao.",
    "Students ke liye part time job.",
    "Instant salary job available.",
    "No interview job apply now.",
    "Earn money from your mobile.",
    "Simple online work available.",
    "Immediate hiring for online workers.",
    "Daily payment job available.",
    "Online data entry work from home.",
    "Job offer limited seats only.",
    "Registration fee pay karo aur job start karo."
]


# ============================================================
# LOTTERY / PRIZE SCAM
# ============================================================

lottery_templates = [

    "Congratulations! You have won 25 lakh.",
    "You have won a lottery prize.",
    "Congratulations you are today's lucky winner.",
    "You won a brand new iPhone.",
    "Your mobile number has won a prize.",
    "Claim your lottery prize immediately.",
    "KBC lottery winner announcement.",
    "You have been selected for a cash prize.",
    "Google lottery winner notification.",
    "You won a free gift voucher.",
    "Prize money is waiting for you.",
    "Congratulations! Claim your reward now.",
    "Your number was selected for the grand prize.",
    "Lucky draw winner. Claim today.",
    "You have won 10 lakh rupees.",
    "Aap lottery winner ban gaye ho.",
    "Aapne 25 lakh rupaye jeete hain.",
    "Congratulations aapka number select hua hai.",
    "Free iPhone prize claim karo.",
    "Lucky draw ka prize aapka hai.",
    "Aapko cash reward mila hai.",
    "Prize lene ke liye details submit karo.",
    "KBC se aapko lottery prize mila hai.",
    "Your phone number has won a cash reward.",
    "You are selected as today's winner.",
    "Claim your free gift now.",
    "Your lucky number has won a prize.",
    "Grand prize winner announcement.",
    "Cash prize waiting for you.",
    "Congratulations! Your reward is ready."
]


# ============================================================
# UPI SCAM
# ============================================================

upi_templates = [

    "Your UPI account has been blocked.",
    "UPI verification required immediately.",
    "Accept this payment request to receive money.",
    "UPI transaction failed. Approve the request.",
    "Payment pending. Confirm the UPI request.",
    "Your UPI ID will be suspended.",
    "UPI blocked. Verify your account.",
    "Collect request pending. Accept now.",
    "UPI payment failed. Click to retry.",
    "Approve the UPI request to receive refund.",
    "Aapka UPI band ho jayega.",
    "UPI verify karo abhi.",
    "Payment receive karne ke liye request accept karo.",
    "UPI transaction complete karne ke liye approve karo.",
    "UPI account verification pending.",
    "UPI payment approve karo jaldi.",
    "UPI request pending hai accept karo.",
    "Aapka UPI blocked hai verify karo.",
    "Payment receive karne ke liye collect request accept karo.",
    "UPI service suspend hone wali hai.",
    "UPI ID verify karna compulsory hai.",
    "Payment failed click here to retry.",
    "UPI refund receive karne ke liye request approve karo.",
    "UPI account block ho jayega verify now.",
    "UPI transaction pending approve immediately.",
    "Collect payment request accept karo.",
    "UPI verification incomplete hai.",
    "UPI payment ke liye approval required hai.",
    "UPI request approve karo warna payment cancel.",
    "UPI service activate karne ke liye verify karo."
]


# ============================================================
# SAFE MESSAGES
# ============================================================

safe_templates = [

    # Greetings
    "Hello",
    "Hi",
    "Hey",
    "Good morning",
    "Good afternoon",
    "Good evening",
    "How are you?",
    "How are you doing?",
    "What are you doing?",
    "Hope you are doing well.",
    "Have a nice day.",
    "Take care.",
    "Good night.",
    "See you soon.",
    "Talk to you later.",

    # Friends
    "Bhai kaha hai?",
    "Bro where are you?",
    "Bhai call kar.",
    "Kal milte hain.",
    "Aaj milte hain.",
    "Khana kha liya?",
    "Kya kar raha hai?",
    "Kab free hai?",
    "Call me when you are free.",
    "Are you free today?",
    "Where are you?",
    "I am on my way.",
    "See you in the evening.",
    "Let's meet tomorrow.",
    "I will call you later.",

    # College
    "Can you send me the notes?",
    "Please send the assignment.",
    "Did you complete the project?",
    "The class starts at 9 am.",
    "The lecture has started.",
    "What time is the class?",
    "Which classroom are we in?",
    "Please share the timetable.",
    "I will reach college by 10.",
    "The professor is coming.",
    "The project deadline is Friday.",
    "Let's discuss the project tomorrow.",
    "I completed the assignment.",
    "Can you send the project file?",
    "The presentation is ready.",
    "Please review the report.",
    "Bhai notes bhej dena.",
    "Kal college aa raha hai?",
    "Assignment complete kar liya?",
    "Project complete ho gaya.",

    # Family
    "Mom is calling.",
    "Dinner is ready.",
    "I am coming home.",
    "We are going out for dinner.",
    "Can you buy some milk?",
    "What should we have for dinner?",
    "I will be home soon.",
    "Dad will reach home later.",
    "Let's have dinner together.",
    "Call me when you reach home.",

    # Travel
    "I booked the tickets.",
    "The train is at 6 pm.",
    "I will meet you at the metro station.",
    "Send me the address.",
    "Can you share the location?",
    "What time are we leaving?",
    "The cab has arrived.",
    "I am waiting at the station.",
    "We will reach by evening.",
    "Let's leave at 7 tomorrow.",

    # Normal payments
    "I received the payment.",
    "Payment received, thank you.",
    "I will pay you tomorrow.",
    "I transferred the rent.",
    "The electricity bill is paid.",
    "I paid the electricity bill.",
    "Please send me the receipt.",
    "Can you send the invoice?",
    "I will transfer the money later.",
    "The payment has been completed.",

    # Work
    "Please check your email.",
    "I sent you the document.",
    "The report looks good.",
    "I will send the file tomorrow.",
    "Can you review this?",
    "The presentation is ready.",
    "Let's discuss this tomorrow.",
    "The meeting is scheduled for Monday.",
    "Please join the meeting at 10.",
    "I have completed the task.",

    # Casual
    "Let's go for lunch.",
    "Want to grab some coffee?",
    "Are you coming to the party?",
    "What movie are we watching?",
    "Let's play football this evening.",
    "I will join you later.",
    "What are your plans for Sunday?",
    "Let's plan something for the weekend.",
    "The weather is really nice today.",
    "I am feeling tired today.",

    # General
    "Thanks for your help.",
    "Thank you.",
    "No problem.",
    "Okay.",
    "Alright.",
    "Sure.",
    "Sounds good.",
    "Congratulations.",
    "Happy birthday.",
    "All the best.",
    "Good luck.",
    "Please call me later.",
    "Can you help me with this?",
    "I will send it shortly.",
    "Let's talk tomorrow."
]


# ============================================================
# DATA GENERATION FUNCTION
# ============================================================

def generate_unique_samples(template_list, label, count):
    """
    Generates unique samples from templates.
    Prevents the dataset from being filled with exact duplicates.
    """

    generated = set()

    attempts = 0
    max_attempts = count * 50

    while len(generated) < count and attempts < max_attempts:

        template = random.choice(template_list)

        text = template.format(
            bank=random.choice(banks),
            app=random.choice(payment_apps),
            urgency=random.choice(urgency),
            hinglish_urgency=random.choice(hinglish_urgency)
        )

        text = text.strip()

        if text:
            generated.add(text)

        attempts += 1

    for text in generated:
        texts.append(text)
        labels.append(label)


# ============================================================
# GENERATE BALANCED DATASET
# ============================================================

generate_unique_samples(
    phishing_templates,
    "Phishing",
    SAMPLES_PER_CLASS
)

generate_unique_samples(
    otp_templates,
    "OTP Scam",
    SAMPLES_PER_CLASS
)

generate_unique_samples(
    kyc_templates,
    "KYC Scam",
    SAMPLES_PER_CLASS
)

generate_unique_samples(
    refund_templates,
    "Refund Scam",
    SAMPLES_PER_CLASS
)

generate_unique_samples(
    job_templates,
    "Fake Job",
    SAMPLES_PER_CLASS
)

generate_unique_samples(
    lottery_templates,
    "Lottery Scam",
    SAMPLES_PER_CLASS
)

generate_unique_samples(
    upi_templates,
    "UPI Scam",
    SAMPLES_PER_CLASS
)

generate_unique_samples(
    safe_templates,
    "Safe",
    SAMPLES_PER_CLASS
)


# ============================================================
# SHUFFLE DATASET
# ============================================================

combined = list(zip(texts, labels))

random.shuffle(combined)

texts, labels = zip(*combined)

texts = list(texts)
labels = list(labels)


# ============================================================
# SHOW DATASET INFORMATION
# ============================================================

from collections import Counter

print("\n==============================================")
print("SAFE LINK DATASET")
print("==============================================")

print(f"Total examples: {len(texts)}")

print("\nClass distribution:")

for category, count in sorted(Counter(labels).items()):
    print(f"{category:15} : {count}")

print("==============================================\n")


# ============================================================
# TF-IDF FEATURE EXTRACTION
# ============================================================
#
# We use BOTH:
#
# 1. WORD N-GRAMS
#    Good for phrases such as:
#    "share otp"
#    "kyc pending"
#    "work from home"
#
# 2. CHARACTER N-GRAMS
#    Good for:
#    "otpp"
#    "kyc"
#    spelling mistakes
#    Hinglish
#    shortened SMS language
#
# ============================================================

word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    min_df=1,
    sublinear_tf=True,
    max_features=15000
)

char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=1,
    sublinear_tf=True,
    max_features=15000
)


# Combine word + character features
vectorizer = FeatureUnion([
    ("word", word_vectorizer),
    ("char", char_vectorizer)
])


# Transform dataset
X = vectorizer.fit_transform(texts)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    labels,
    test_size=0.20,
    stratify=labels,
    random_state=42
)


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

model = LogisticRegression(
    max_iter=3000,
    class_weight="balanced",
    C=3.0,
    solver="liblinear"
)


# ============================================================
# TRAIN
# ============================================================

print("Training SafeLink AI model...\n")

model.fit(X_train, y_train)


# ============================================================
# MODEL EVALUATION
# ============================================================

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("==============================================")
print("MODEL EVALUATION")
print("==============================================")

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# ============================================================
# TEST SAMPLE MESSAGES
# ============================================================

test_messages = [

    # SAFE
    "Hello",
    "Hi bro how are you?",
    "Can you send me the notes?",
    "Meeting at 5 pm today",
    "Bhai kal college aa raha hai?",
    "I received the payment, thank you.",

    # PHISHING
    "Your SBI account will be blocked. Verify immediately.",
    "Suspicious login detected, confirm your account now.",

    # OTP
    "Apna OTP bhejo warna account band ho jayega",
    "Sir OTP confirm kar do jaldi",

    # KYC
    "Aapka KYC expire ho gaya hai update karo",
    "KYC pending hai warna UPI band ho jayega",

    # REFUND
    "Your refund is pending, send your bank details.",
    "Refund ke liye account details bhejo",

    # JOB
    "Work from home earn 5000 daily",
    "Bhai online job hai daily payment milega",

    # LOTTERY
    "Congratulations you won 25 lakh rupees",
    "Aap lottery winner ban gaye ho claim karo",

    # UPI
    "UPI blocked verify now",
    "Payment receive karne ke liye request accept karo"
]


print("\n==============================================")
print("LIVE MODEL TEST")
print("==============================================\n")


for message in test_messages:

    x = vectorizer.transform([message])

    prediction = model.predict(x)[0]

    probabilities = model.predict_proba(x)[0]

    confidence = probabilities.max() * 100

    print(f"Message    : {message}")
    print(f"Prediction : {prediction}")
    print(f"Confidence : {confidence:.2f}%")
    print("----------------------------------------------")


# ============================================================
# SAVE MODEL + VECTORIZER
# ============================================================

model_path = os.path.join(
    BASE_DIR,
    "scam_model.pkl"
)

vectorizer_path = os.path.join(
    BASE_DIR,
    "vectorizer.pkl"
)


joblib.dump(
    model,
    model_path
)

joblib.dump(
    vectorizer,
    vectorizer_path
)


# ============================================================
# COMPLETE
# ============================================================

print("\n==============================================")
print("✅ SafeLink AI MODEL TRAINED SUCCESSFULLY")
print("==============================================")

print(f"Model saved to:")
print(model_path)

print(f"\nVectorizer saved to:")
print(vectorizer_path)

print("\nRestart your Flask backend after training.")
print("==============================================")