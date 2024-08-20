import sys

def error_message_detail(error: Exception, error_detail: sys) -> str:
    """
    Creates a detailed error message including file name, line number, and error message.
    """
    _, _, exc_tb = error_detail.exc_info()
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        error_message = (
            f"Error occurred in script: {file_name}, line number: {line_number}, error message: {str(error)}"
        )
    else:
        error_message = str(error)
    
    return error_message

class CustomException(Exception):
    def __init__(self, error_message: Exception, error_detail: sys):
        """
        Initializes the CustomException with a formatted error message.
        """
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self) -> str:
        return self.error_message
