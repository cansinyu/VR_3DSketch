from flask import Flask, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # 启用 CORS

@app.route('/get_fbx')
def get_fbx():
    fbx_path = '/root/autodl-tmp/project/VR/input/1.FBX'  # 替换为你的 FBX 文件在服务器上的路径
    try:
        return send_file(fbx_path, as_attachment=True)
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    port = 8443
    print(f"Starting Flask server on http://0.0.0.0:{port}")
    print("Press Ctrl+C to stop the server")
    app.run(host='0.0.0.0', port=port)
#    app.run()
