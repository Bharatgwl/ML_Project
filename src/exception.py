import sys
from src.logger import logging
def error_message_details(error, erro_detail: sys):
    _, _, exc_tb = erro_detail.exc_info()
    error_message = (
        "Error: [ "
        + str(error)
        + " ] in file [ "
        + str(exc_tb.tb_frame.f_code.co_filename)
        + " ] line number [ "
        + str(exc_tb.tb_lineno)
        + " ] "
        + str(error)
        + "\n"  
    )
    return error_message
class CustomException(Exception):
    def __init__(self,error_message,erro_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message,erro_detail=erro_detail)
    def __str__(self):
        return self.error_message

# if __name__ == "__main__":
#     try:
#         a = 1/0
#     except Exception as e:
#         logging.info("Divide by zero")
#         raise CustomException(e,sys)