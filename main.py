from flask import Flask, request, Response
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
from service import service_handling
from openai_pdf_chat.pdf_chat import main
import requests
from PIL import Image
from io import BytesIO
from voice_recognition.handle_voice import voice_handler
import unicodedata
import os

app = Flask(__name__)
app.config["SECRET_KEY"] = "top-secret!"  # SECRET KEY CAN BE ANYTHING

account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
auth_token = os.environ.get("TWILIO_AUTH_TOKEN")

client = Client(account_sid, auth_token)


def send_message(body_mess, phone_number):
    message = client.messages.create(
        from_="whatsapp:+14155238886",  # With Country Code
        body=body_mess,
        to="whatsapp:" + phone_number,  # With Country Code
    )
    print(message)


def send_media_message(media_url, phone_number):
    client.messages.create(
        from_="whatsapp:+14155238886",  # With Country Code
        media_url=[media_url],
        to="whatsapp:" + phone_number,  # With Country Code
    )


welcoming_message = 0


@app.route("/bot", methods=["POST"])
def bot():
    global welcoming_message

    # Debug logs
    print("Received a message:")
    print(request.values)

    phone_number = request.values["WaId"]

    if welcoming_message == 0:
        send_message(
            "أهلاً بك في Ai HukoomiBot\n\nالخدمات الحكومية التي أقدمها  :\n\n1. خدمة الاستعلام عن مخالفات السير: رمز الخدمة هو 11\n2. خدمة الذمم المالية لوزارة المالية: رمز الخدمة هو 12.\n3. خدمة استعلام قيم مخالفات السير حسب قانون السير: رمز الخدمة هو 13.\n4. خدمة ضريبة الأبنية والمسقفات: رمز الخدمة هو 14.\n5. خدمة الاستعلام عن قطعة الأرض: رمز الخدمة هو 15.\n6. خدمة معلومات مشترك الضمان الاجتماعي: رمز الخدمة هو 16.\n7. خدمة الترتيب التنافسي لديوان الخدمة: رمز الخدمة هو 17.\n8. خدمة كتاب لمن يهمه الأمر من الضمان الاجتماعي: رمز الخدمة هو 18.\n9. خدمة متابعة طلبات الأحوال المدنية: رمز الخدمة هو 19.\n10. خدمة إصدار شهادة القيد العائلي: رمز الخدمة هو 20.\n11. خدمة طلب براءة الذمة من أمانة عمان: رمز الخدمة هو 21.\n\nيرجى ملاحظة أنه لتنفيذ الخدمة المرغوبة، يجب إدخال رمز الخدمة فقط.\n\nيمكنك أيضاً الاستعلام والسؤال عن أي معلومة تخص أي دائرة من الدوائر الحكومية.",
            phone_number,
        )
        welcoming_message += 1
        return ""

    if request.values["Body"]:
        incoming_msg = request.values["Body"]

        # Print the incoming message here
        print("Sent message:")
        print(incoming_msg)

    elif (
        request.values["MediaUrl0"]
        and request.values["MediaContentType0"] == "audio/ogg"
    ):
        incoming_msg = request.values["MediaUrl0"]
        response = requests.get(incoming_msg)
        audio_content = response.content
        with open("./voice_message/audio_test.ogg", "wb") as file:
            file.write(audio_content)
        incoming_msg = voice_handler()

        print(incoming_msg)

        normalized_text = unicodedata.normalize("NFKC", incoming_msg)

        # Convert the normalized text to standard Arabic Unicode range
        converted_text = "".join(
            char for char in normalized_text if unicodedata.bidirectional(char) == "AL"
        )

        incoming_msg = converted_text
        print(converted_text)

    elif (
        request.values["MediaUrl0"]
        and request.values["MediaContentType0"] == "image/jpeg"
        and service_handling.current_process == "11"
    ):
        incoming_msg = request.values["MediaUrl0"]
        response = requests.get(incoming_msg)
        image_content = response.content
        image = Image.open(BytesIO(image_content))
        image.save(
            f"./images/user_image_input{len(list(service_handling.responses.values())[0]) + 1}.jpeg"
        )

    else:
        send_message("Message Cannot Be Empty!", phone_number)

    try:
        service_handling.service(incoming_msg, phone_number)
    except Exception as e:
        print("Error in service_handling.service:", str(e))
        answer = main(incoming_msg)
        send_message(answer, phone_number)
        print("Answer from main:", answer)
    r = MessagingResponse()
    r.message("")
    return str(r)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=os.environ.get("PORT", 5000))
