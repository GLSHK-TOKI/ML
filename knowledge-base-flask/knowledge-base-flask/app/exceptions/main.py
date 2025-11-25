class BackendError(Exception):
    def __init__(self, message):
        self.message = message
        self.status_code = 500

class BadRequestError(BackendError):
    def __init__(self, message):
        self.message = message
        self.status_code = 400

class UnauthorizedError(BackendError):
    def __init__(self, message):
        self.message = message
        self.status_code = 403