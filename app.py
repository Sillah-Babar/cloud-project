from distutils.log import debug
from fileinput import filename
from flask import *
import boto3

app = Flask(__name__)



# AWS S3 configuration
S3_BUCKET = 'aws-project-bucket-2'
S3_ACCESS_KEY = 'AKIAQ5YBX4K7LHSGGDCZ'
S3_SECRET_KEY = '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'
S3_LOCATION = 'http://{}.s3.amazonaws.com/'.format(S3_BUCKET)

s3 = boto3.client(
    's3',
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY
)

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    s3.upload_fileobj(
        file,
        S3_BUCKET,
        file.filename,
        ExtraArgs={
            "ACL": "public-read",
            "ContentType": file.content_type
        }
    )
    return 'File uploaded successfully to S3!'


@app.route('/')
def main():
	return render_template("index.html")

@app.route('/success', methods = ['POST'])
def success():
	if request.method == 'POST':
		f = request.files['file']
		f.save(f.filename)
		return render_template("Acknowledgement.html", name = f.filename)

if __name__ == '__main__':
	app.run(debug=True)
