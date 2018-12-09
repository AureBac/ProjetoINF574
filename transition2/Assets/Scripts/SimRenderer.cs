using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using System.Runtime.InteropServices;
using Unity.Mathematics;

public class SimRenderer : MonoBehaviour {
    public Mesh instanceMesh;
    [SerializeField] private Material instanceMaterial;

    private Main sim_controller;

    // Compute buffers used for indirect mesh drawing
    private ComputeBuffer point_buffer;
    private ComputeBuffer args_buffer;
    private uint[] args = new uint[5] { 0, 0, 0, 0, 0 };

    private Bounds bounds;
    // Use this for initialization
    void Start () {
        sim_controller = GameObject.FindObjectOfType<Main>();

        // Create a compute buffer that holds (# of particles), with a byte offset of (size of particle) for the GPU to process
        point_buffer = new ComputeBuffer(sim_controller.ps.Length, Marshal.SizeOf(new Particle()), ComputeBufferType.Default);

        // ensure the material has a reference to this consistent buffer so the shader can access it
        instanceMaterial.SetBuffer("pos_buffer", point_buffer);

        // indirect arguments for mesh instances
        args_buffer = new ComputeBuffer(1, args.Length * sizeof(uint), ComputeBufferType.IndirectArguments);
        uint numIndices = (uint)instanceMesh.GetIndexCount(0);
        args[0] = numIndices;
        args[1] = (uint)sim_controller.ps.Length;
        args_buffer.SetData(args);

        // create rendering bounds for DrawMeshInstancedIndirect
        bounds = new Bounds(Vector3.zero, new Vector3(100, 100, 100));

        //point_buffer.SetData(sim_controller.ps);

        //Graphics.DrawMeshInstancedIndirect(instanceMesh, 0, instanceMaterial, bounds, args_buffer);
        
    }
	
	// Update is called once per frame
	void Update () {
        point_buffer.SetData(sim_controller.ps);

        Graphics.DrawMeshInstancedIndirect(instanceMesh, 0, instanceMaterial, bounds, args_buffer);
    }

    void OnDisable()
    {
        if (args_buffer != null) args_buffer.Release();
        if (point_buffer != null) point_buffer.Release();
    }
}
