using System.Collections;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.UI;
using System.IO;
using TriLibCore;
using TMPro;
using System.Collections.Generic;

public static class GlobalAssetLoaderContext
{
    public static List<AssetLoaderContext> assetLoaderContexts = new List<AssetLoaderContext>();
}

public class CountDownScript : MonoBehaviour
{
    [SerializeField] private Text uiText;

    private float timer;
    private bool canCount = false;
    public bool doOnce = false;
    private SaveSketchLogic saveSketchLogic;  // Assuming you have an instance of SaveSketchLogic
    public Transform boxCenterTrf;//正方体子物体中心空物体
    public Slider slider;
    public Material mat;
    

    
    void Start()
    {
        timer = 10.0f;
        // 初始化按钮对象
        Button button = GetComponent<Button>();
        if (button != null)
        {
            // 添加按钮点击事件监听器
            button.onClick.AddListener(OnClickButton);
        }
        else
        {
            Debug.LogError("Button component not found on the game object.");
        }
        slider.transform.GetChild(3).gameObject.SetActive(false);
    }

    void Update()
    {

    }

    public void init()
    {
        canCount = false;
        doOnce = false;
        timer = 10.0f;
    }

    public void StartCount()
    {
        // 重置进度条 UI 为初始状态
        slider.value = 0;
        slider.transform.GetChild(1).GetChild(0).GetComponent<Image>().color = Color.red;

        // 重置文本为默认值，并确保相关 UI 组件处于激活状态
        slider.transform.GetChild(3).GetComponent<Text>().text = "Building the magic...";
        slider.transform.GetChild(3).gameObject.SetActive(true);
        slider.transform.GetChild(3).GetChild(0).gameObject.SetActive(true);

        // 开始发送 POST 请求
        StartCoroutine(SendPostRequest());
    }

    private void OnClickButton()
    {
        StartCoroutine(SendPostRequest());
    }

    private IEnumerator SendPostRequest()
    {
 
                // 创建一个 WWWForm 对象
                WWWForm form = new WWWForm();

                // 获取根目录下的 txt 和 off 文件路径列表
                string rootDirectory = @".\path\demo_savedir";  // 这里使用 Application.dataPath 作为根目录，你可以根据实际情况修改
                //string rootDirectory = @"C:\3D\dww\Sketch_VR\Assets\path\demo_savedir";  // 这里使用 Application.dataPath 作为根目录，你可以根据实际情况修改
                string[] txtFiles = Directory.GetFiles(rootDirectory, "*.txt");
                string[] offFiles = Directory.GetFiles(rootDirectory, "*.off");

                // 遍历 txt 文件列表，添加到 form 中
                foreach (string txtFilePath in txtFiles)
                {
                    // 获取文件名和文件字节数组
                    string txtFileName = Path.GetFileName(txtFilePath);
                    byte[] txtFileBytes = File.ReadAllBytes(txtFilePath);

                    // 添加到 WWWForm
                    form.AddBinaryData("txt", txtFileBytes, txtFileName, "text/plain");
                    form.AddField("txt_filename", txtFileName);  // 添加 txt 文件名字段
                }

                // 遍历 off 文件列表，添加到 form 中
                foreach (string offFilePath in offFiles)
                {
                    // 获取文件名和文件字节数组
                    string offFileName = Path.GetFileName(offFilePath);
                    byte[] offFileBytes = File.ReadAllBytes(offFilePath);

                    // 添加到 WWWForm
                    form.AddBinaryData("off", offFileBytes, offFileName, "text/plain");
                    form.AddField("off_filename", offFileName);  // 添加 off 文件名字段
                }

                // 使用 UnityWebRequest 发送 POST 请求
                using (UnityWebRequest www = UnityWebRequest.Post("http://192.168.50.57:6006/get_fbx", form))
                {
                    yield return www.SendWebRequest();

                    if (!www.isNetworkError && !www.isHttpError)
                    {


                        // 将后端返回的数据保存为FBX文件
                        string fbxFilePath = Path.Combine(rootDirectory, "received_model.fbx");

                        File.WriteAllBytes(fbxFilePath, www.downloadHandler.data);


                        //模型加载进度条显示
                        slider.value = 0;
                        slider.transform.GetChild(1).GetChild(0).GetComponent<Image>().color = Color.red;


                        // 使用 TriLib 加载 FBX 模型
                        var assetLoaderOptions = AssetLoader.CreateDefaultLoaderOptions();
                        AssetLoader.LoadModelFromFile(fbxFilePath, OnLoad, OnMaterialsLoad, OnProgress, OnError, null, assetLoaderOptions);
                    }
                    else
                    {
                        Debug.Log("Request failed: " + www.error);
                    }
                }




    }
    private void OnLoad(AssetLoaderContext assetLoaderContext)
    {
        Debug.Log("加载的模型名称" + assetLoaderContext.RootGameObject.name);

        // 设置模型的父亲对象（立方体）
        assetLoaderContext.RootGameObject.transform.SetParent(boxCenterTrf);
        // 设置位置为中心点
        assetLoaderContext.RootGameObject.transform.localPosition = Vector3.zero;
        // 设置模型的旋转与立方体的旋转相同
        assetLoaderContext.RootGameObject.transform.rotation = boxCenterTrf.rotation;

        assetLoaderContext.RootGameObject.transform.localScale = boxCenterTrf.localScale;
        GlobalAssetLoaderContext.assetLoaderContexts.Add(assetLoaderContext);
        if (mat != null)
        {
            //assetLoaderContext.RootGameObject.transform.GetComponent<Renderer>().material = mat;
            for (int i = 0; i < assetLoaderContext.RootGameObject.transform.childCount; i++)
            {
                assetLoaderContext.RootGameObject.transform.GetChild(i).GetComponent<Renderer>().material = mat;
            }
        }
        // 在模型加载后开始协程以隐藏或销毁模型
       // StartCoroutine(HideModelAfterDelay(assetLoaderContext.RootGameObject, 3f));

    }

    private void OnMaterialsLoad(AssetLoaderContext assetLoaderContext)
    {
        Debug.Log("加载完成");
        slider.transform.GetChild(1).GetChild(0).GetComponent<Image>().color = Color.green;
        slider.transform.GetChild(3).GetComponent<Text>().text = "Done!!!";
        slider.transform.GetChild(3).GetChild(0).gameObject.SetActive(false);
    }
    private void OnProgress(AssetLoaderContext assetLoaderContext, float progress)
    {
        Debug.Log($"正在加载模型，进度为: {progress:P}");
        slider.value = progress;
    }
    private void OnError(IContextualizedError obj)
    {
        Debug.LogError($"An error occurred while loading your Model: {obj.GetInnerException()}");
    }
}
