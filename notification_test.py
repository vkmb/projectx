from pyfcm import FCMNotification
import smtplib 
  
# creates SMTP session 
s = smtplib.SMTP('smtp.gmail.com', 587) 
  
# start TLS for security 
s.starttls() 
  
# Authentication 
s.login(, ) 
  
# message to be sent 
message = "Message_you_need_to_send"
  
# sending the mail 
s.sendmail("", "", message) 
  
# terminating the session 
s.quit() 



# push_service = FCMNotification(api_key="AAAAya_HRmk:APA91bHHeZdiyjgmeHGh3NuQvTcY5H-uY98a5jojbcFHJXttebAR8COfBDMopevY6S6TtpI3ItFbXwd1HnyXbK0I6ipBMgPfKLw3q0CzS-oWv3Mk3hfN0vXUY1BG6s0ipXAY73OzSyM7")
# while input():
# 	result = push_service.notify_topic_subscribers(topic_name="GSR8uoMXkPMqDRdMnWqlYVLZlBr1", message_body="intruder")
# 	print(result['success'])