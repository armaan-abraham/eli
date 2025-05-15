import logging
import boto3
import time
from io import StringIO
from botocore.exceptions import ClientError

class S3LogHandler(logging.Handler):
    def __init__(self, bucket, key_prefix, flush_interval=60):
        """Create an S3 log handler that aggregates all logs for the current
        Python process in a *single* S3 object.  A unique key is generated once
        (based on the current timestamp) and reused on every flush, so new log
        lines are appended to the same remote file instead of creating a new
        object per flush.

        Args:
            bucket (str): Name of the S3 bucket.
            key_prefix (str): Prefix under which the log object should be
                stored. The handler will create the object at
                ``{key_prefix}/logs/{timestamp}.log`` where *timestamp* is the
                moment the handler is instantiated.
            flush_interval (int): Flush interval in seconds.
        """
        super().__init__()
        self.bucket = bucket
        self.key_prefix = key_prefix
        self.buffer = StringIO()
        self.s3_client = boto3.client('s3')
        self.last_flush = time.time()
        self.flush_interval = flush_interval
        
        # Create a deterministic key for the lifetime of this handler.  We take
        # a timestamp at instantiation time to make sure concurrent runs don't
        # collide with each other while still aggregating all writes from the
        # current run into a single object.
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.log_key = f"{self.key_prefix}/logs/{timestamp}.log"
        
    def emit(self, record):
        msg = self.format(record)
        self.buffer.write(msg + '\n')
        
        # Periodically flush to S3
        if time.time() - self.last_flush > self.flush_interval:
            self.flush()
        
    def flush(self):
        if self.buffer.tell() == 0:
            return

        # Prepare buffer for reading
        self.buffer.seek(0)

        try:
            # Try to retrieve the existing log object so we can append the new
            # data to it.  If the object does not exist yet (first flush), we
            # treat it the same as an empty object.
            try:
                response = self.s3_client.get_object(Bucket=self.bucket, Key=self.log_key)
                existing_data = response["Body"].read()
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")
                # If the object does not exist yet, we just start with an empty
                # byte string; otherwise, re-raise the error.
                if error_code == "NoSuchKey":
                    existing_data = b""
                else:
                    raise

            # Append the new log lines we have just collected.
            new_data = existing_data + self.buffer.getvalue().encode("utf-8")

            # Upload the combined object back to S3 *overwriting* the existing one.
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=self.log_key,
                Body=new_data,
                ContentType="text/plain",
            )

            # Reset buffer and flush timer
            self.buffer = StringIO()
            self.last_flush = time.time()
        except Exception as e:
            # Log error but don't stop execution. The user may have another
            # handler (e.g. console) attached, so using print is acceptable
            # here.
            print(f"Failed to upload logs to S3: {str(e)}")
        
    def close(self):
        self.flush()
        super().close()

def configure_logging(bucket, key_prefix):
    """
    Configure logging to both console and S3.
    
    Args:
        bucket: S3 bucket name
        key_prefix: Prefix for log files in S3
        
    Returns:
        S3LogHandler instance for later cleanup
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    
    # Add S3 handler
    s3_handler = S3LogHandler(bucket, key_prefix)
    s3_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(s3_handler)
    
    return s3_handler

def log_progress(current, total):
    """
    Log progress information.
    
    Args:
        current: Current number of samples processed
        total: Total number of samples to process
    """
    percentage = (current / total) * 100 if total > 0 else 0
    logging.getLogger().info(f"Progress: {current}/{total} samples processed ({percentage:.2f}%)") 