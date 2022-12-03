import RPi.GPIO as GPIO
rebort=7
while True:
    if GPIO.input(rebort) == GPIO.LOW:
        print("未接通")
    else:
        print("以接通")