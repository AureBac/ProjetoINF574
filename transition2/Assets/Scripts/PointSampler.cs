using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Mathematics;
//using Unity.Jobs;
//using Unity.Burst;


public class PointSampler : MonoBehaviour {
    [SerializeField] private Mesh instanceMesh;


    [HideInInspector] public List<float3> points = new List<float3>();
    [HideInInspector] public List<float> masses = new List<float>();

    public GameObject selector;
    public GameObject[] arr;
    // Use this for initialization
    void Start () {
        Vector3[] v = instanceMesh.vertices;
        //int n = 2;
        int n = v.Length;

        //selector = this.transform.parent.gameObject;
        //arr = new GameObject[n];
        //GetComponent<Camera>().
        for (int i=0; i< n; i++)
        {
          points.Add(v[i]);
          masses.Add(0.5f);


            if (i == 1)
            {
                //GameObject go = GameObject.CreatePrimitive(PrimitiveType.Cube);
                //go.transform.position = v[i];
                //go.transform.localScale = new Vector3(0.5f, 0.5f, 0.5f);
            }
            //GameObject go = GameObject.CreatePrimitive(PrimitiveType.Cube);
            //go.transform.position = v[i];
            //go.transform.localScale = new Vector3(0.01f, 0.01f, 0.01f);
            //float2 a = math.float2(5.0f, 5.0f) ;
            //float2 b = a.xy;
            //arr[i]=go;
            //arr[i].transform.parent = selector.transform;
        }
        //points.Add(v[0]);
        //masses.Add(0.5f);
        Debug.Log(v[0]);
        Debug.Log(v.Length);
	}
	
	// Update is called once per frame
	void Update () {

         
}
}
