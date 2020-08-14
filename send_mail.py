from supporting_functions import get_service, create_message, send_message
import datetime
import pickle

message_text = "Automated mailing list test.  From Marshall's server."

path_to_pickle = "keys\\gmail\\token.pickle"
subject = "Financial News and Advice | {}".format(datetime.date.today())
sender = "YOUR_MAIL_ID"
user_id = "me"

with open('test_mailing_list.pkl', 'rb') as f:
    mail_dict = pickle.load(f)

for user in mail_dict.keys():
    mail_to = mail_dict[user]['email']
    try:
        service = get_service(path=path_to_pickle)
        raw_text = create_message(sender=sender, to=mail_to, subject=subject, message_text=message_text)
        message_data = send_message(service=service, user_id=user_id, message=raw_text)
        print('Mailed to {} successfully!'.format(mail_to))
    except Exception as e:
        print('Encountered error {} mailing to {}!'.format(e, mail_to))

# need to uncomment the build import in supporting_functions.py
