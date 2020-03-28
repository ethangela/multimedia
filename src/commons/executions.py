import os
import logging
import time

# TODO: Have a centralized logging object !!!!
logging.basicConfig(level=logging.INFO)

# TODO: No hardcoding please ...
_ITERATIONS 	= 5
_DELAY 			= 2

def multiple_executions_wrapper(fnct):
	def inner(*args, **kwargs):
		_ex 	= None
		for _ in range(_ITERATIONS):
			try:
				resp = fnct(*args, **kwargs)
				return resp
			except Exception as ex:
				time.sleep(_DELAY)
				_ex = ex
				logging.warning("PID: {0} Exception occured: {1}".format(os.getpid(), ex))
				logging.warning("PID: {0} attempting retry".format(os.getpid()))
		
		if _ex is not None:
			logging.error("PID: {0} terminating with exception: {1}".format(os.getpid(), _ex))
			raise _ex
	return inner