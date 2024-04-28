using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ClearModel : MonoBehaviour
{
    private void Start()
    {
        // 初始化代码（如果有的话）
    }

    public void Clear()
    {
        // 检查是否有模型可以清除
        if (GlobalAssetLoaderContext.assetLoaderContexts.Count > 0 && GlobalAssetLoaderContext.assetLoaderContexts[0].RootGameObject != null)
        {
            int length = GlobalAssetLoaderContext.assetLoaderContexts.Count;
            for(int i=0; i<length; i++)
            {
                Destroy(GlobalAssetLoaderContext.assetLoaderContexts[i].RootGameObject);
            }
            GlobalAssetLoaderContext.assetLoaderContexts.Clear();


        }
    }

    public void Quit()
    {
        Application.Quit();
    }
}
