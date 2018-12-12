using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Collections;

using Unity.Jobs;
using UnityEngine.Jobs;
using Unity.Mathematics;
using Unity.Burst;

public struct Particle
{
    public float3 x;
    public float3 p;
    public float3 v;
    public float3 f;
    public float inv_mass;
}

public struct QuadParticle
{
    public float[,] quad;
}

public class Main : MonoBehaviour
{
    float[,] matrix = new float[3, 9];
    //Debug.Log(matrix);

    public NativeArray<Particle> ps;
    //public NativeArray<Particle> psquadratic;

    int num_particles;
    const int division = 128;
    //Debug.Log(division);
    const int mouse_force = 10;
    const float mouse_influence_radius = 1.0f;

    const float gravity = -9.8f;
    [SerializeField]
    private float g = -2.0f;

    //const float gravity = -2.0f;
    const float stiffness = 0.25f;
    //const float stiffness = 1.0f;
    //const float linear_deformation_blending = 0.3f;
    const float linear_deformation_blending = 0.9f;

    const float eps = 1e-6f;
    float dt;
    float3 bounds;
    float3x3 inv_rest_matrix;
    NativeArray<float3> deltas; // precomputed distances of particles to shape center of mass
    NativeArray<float4> com_sums; // temp array for parallelised center of mass summation
    NativeArray<float3x3> shape_matrices; // temp array for parallelised deformed shape matrix summation

    #region Jobs
    #region Integration
    [BurstCompile]
    struct Job_Integrate0 : IJobParallelFor
    {
        public NativeArray<Particle> ps;

        [ReadOnly]
        public float dt;

        //teste de alteração em tempo real
        //public float g;
        //
        public void Execute(int i)
        {
            Particle p = ps[i];
            p.v += p.f * dt;
            p.p += p.v * dt;
            p.f = math.float3(0, gravity, 0);

            //teste tempo real
            //p.f = math.float3(0, g, 0);
            //
            ps[i] = p;
        }
    }

    [BurstCompile]
    struct Job_Integrate1 : IJobParallelFor
    {
        public NativeArray<Particle> ps;

        [ReadOnly]
        public float3 bounds;
        [ReadOnly]
        public float inv_dt;

        public void Execute(int i)
        {
            Particle p = ps[i];
            // Super simple boundary conditions, clamping positions and adding some fake ground friction,
            // by changing the particle's next-frame position to be a little closer to its start position if it penetrated the ground
            if (p.p[1] < -bounds.y) p.p[0] = 0.5f * (p.x[0] + p.p[0]);
            p.p = math.clamp(p.p, -bounds, bounds);
            p.v = (p.p - p.x) * inv_dt;
            p.x = p.p;
            ps[i] = p;


        }
    }

    #endregion

    #region Miscellaneous
    [BurstCompile]
    struct Job_ApplyMouseForce : IJobParallelFor
    {
        public NativeArray<Particle> ps;
        [ReadOnly]
        public float3 cam_point;

        public void Execute(int i)
        {
            Particle p = ps[i];

            // just calculating a directional force and applying it uniformly to all particles
            float3 dist = (ps[i].p - cam_point);
            p.p += math.normalize(dist) * 0.01f;
            ps[i] = p;
        }
    }



    #endregion

    #region Shape Matching Jobs


    [BurstCompile]
    struct Job_SumCenterOfMass : IJobParallelFor
    {
        [WriteOnly]
        public NativeArray<float4> com_sums;
        [ReadOnly]
        public NativeArray<Particle> ps;
        [ReadOnly]
        public int stride;

        public void Execute(int i)
        {
            // only perform this step from the start of each batch index, every stride'th entry in the array onwards
            if (i % stride != 0) return;

            // calculate center of mass for this batch
            float3 cm = math.float3(0);

            float wsum = 0.0f;
            for (int idx = i; idx < i + stride; ++idx)
            {
                Particle p = ps[idx];
                float wi = 1.0f / (p.inv_mass + eps);
                cm += p.p * wi;
                wsum += wi;
            }

            // storing the total weight in the z component for use when combining later
            float4 result = math.float4(cm.x, cm.y, cm.z, wsum);

            com_sums[i] = result;

        }





    }

    [BurstCompile]
    struct Job_SumShapeMatrix : IJobParallelFor
    {
        [WriteOnly]
        public NativeArray<float3x3> shape_matrices;
        [ReadOnly]
        public float3 cm;
        [ReadOnly]
        public NativeArray<Particle> ps;
        [ReadOnly]
        public NativeArray<float3> deltas;
        [ReadOnly]
        public int stride;

        public void Execute(int i)
        {
            // same idea as in center of mass calculation
            if (i % stride != 0) return;

            // this is part of eq. (7) in the original paper, finding the optimal linear transformation matrix between our reference and deformed positions
            float3x3 mat = math.float3x3(0, 0, 0);
            for (int idx = i; idx < i + stride; ++idx)
            {
                Particle pi = ps[idx];
                float3 q = deltas[idx];
                float3 p = pi.p - cm;
                float w = 1.0f / (pi.inv_mass + eps);
                p *= w;

                mat.c0 += p * q[0];
                mat.c1 += p * q[1];
                mat.c2 += p * q[2];
            }

            shape_matrices[i] = mat;



        }

    }
    [BurstCompile]
    struct Job_GetDeltas : IJobParallelFor
    {
        public NativeArray<Particle> ps;

        [ReadOnly]
        public float3 cm;
        [ReadOnly]
        public NativeArray<float3> deltas;
        [ReadOnly]
        public float3x3 GM;

        public void Execute(int i)
        {
            // calculating the "ideal" position of a particle by multiplying by the deformed shape matrix GM, offset by our center of mass
            float3 goal = math.mul(GM, deltas[i]) + cm;

            // amount to move our particle this timestep for shape matching. if stiffness = 1.0 this corresponds to rigid body behaviour
            // (though you'd need to do more PBD iterations in the main loop to get this stiff enough)
            float3 delta = (goal - ps[i].p) * stiffness;

            Particle p = ps[i];
            p.p += delta;
            ps[i] = p;

        }
    }

    #endregion


    #region Shape Matching Jobs Quad
    [BurstCompile]
    struct Job_SumCenterOfMassQuad : IJobParallelFor
    {
        [WriteOnly]
        public NativeArray<float4> com_sums;
        [ReadOnly]
        public NativeArray<Particle> ps;
        //[ReadOnly] public NativeArray<QuadParticle> psquadratic;
        [ReadOnly]
        public int stride;

        float[,] changeToNine(float3 position)
        {
            float[,] result = new float[9, 1];
            result[0, 0] = position.x;
            result[1, 0] = position.y;
            result[2, 0] = position.z;

            result[3, 0] = position.x * position.x;
            result[4, 0] = position.y * position.y;
            result[5, 0] = position.z * position.z;

            result[6, 0] = position.x * position.y;
            result[7, 0] = position.y * position.z;
            result[8, 0] = position.z * position.x;

            return result;
        }

        public void Execute(int i)
        {
            // only perform this step from the start of each batch index, every stride'th entry in the array onwards
            if (i % stride != 0) return;

            // calculate center of mass for this batch
            float3 cm = math.float3(0);
            float[,] cmQuad = new float[9, 1];

            float wsum = 0.0f;
            for (int idx = i; idx < i + stride; ++idx)
            {
                Particle p = ps[idx];
                float wi = 1.0f / (p.inv_mass + eps);
                cm += p.p * wi;
                wsum += wi;
            }
            ////////////////////change
            cmQuad = changeToNine(cm);
            ///////////////////////////
            // storing the total weight in the z component for use when combining later
            float4 result = math.float4(cm.x, cm.y, cm.z, wsum);

            com_sums[i] = result;

        }





    }



    #endregion

    #endregion


    #region Math Functions

    //////////////////////////////////////////////////////////////////////////
    // MathFunctions
    //////////////////////////////////////////////////////////////////////////


    //------------------------------------------------------------------------------------------------
    NativeArray<float3x3> jacobiRotate(float3x3 A1, float3x3 R1, int p, int q)
    {
        NativeArray<float3x3> AR = new NativeArray<float3x3>(2, Allocator.Persistent);
        //AR[0] = A1;
        //AR[1] = R1;
        float3x3 A = A1;
        float3x3 R = R1;
        //A[p] = math.float3(0);

        // rotates A through phi in pq-plane to set A(p,q) = 0
        // rotation stored in R whose columns are eigenvectors of A
        //float3 c=math.float3(0);
        /*if (p == 0)
        {
            c = A.c0;
        }
        else if (p == 1)
        {
            c = A.c1;
        }
        else if (p == 2)
        {
            c = A.c2;
        }

        float Apq;
        if (q == 0)
        {
            Apq = c.x;
        }
        else if (q == 1)
        {
            Apq = c.y;
        }
        else if (q == 2)
        {
            Apq = c.z;
        }*/

        if (A[p][q] == 0.0f)
        {
            AR[0] = A1;
            AR[1] = R1;
            return AR;
        }
        float d = (A[p][p] - A[q][q]) / (2.0f * A[p][q]);
        float t = (1.0f) / (math.abs(d) + math.sqrt(d * d + 1.0f));
        if (d < 0.0f)
        {
            t = -t;
        }
        float c = (1.0f) / math.sqrt(d * d + 1.0f);
        float s = t * c;
        float3 As = A[p];
        As[p] += t * A[p][q];
        A[p] = As;
        As = A[q];
        As[q] -= t * A[p][q];
        A[q] = As;
        As = A[p];
        As[q] = 0.0f;
        A[p] = As;
        As = A[q];
        As[p] = 0.0f;
        A[q] = As;

        //transform A
        int k;

        for (k = 0; k < 3; k++)
        {
            if (k != p && k != q)
            {
                float Akp = c * A[k][p] + s * A[k][q];
                float Akq = -s * A[k][p] + c * A[k][q];
                As = A[k];
                As[p] = Akp;
                A[k] = As;
                As = A[p];
                As[k] = Akp;
                A[p] = As;

                As = A[k];
                As[q] = Akq;
                A[k] = As;
                As = A[q];
                As[k] = Akq;
                A[q] = As;



            }
        }

        // strore rotation in  R
        for (k = 0; k < 3; k++)
        {
            float Rkp = c * R[k][p] + s * R[k][q];
            float Rkq = -s * R[k][p] + c * R[k][q];
            As = R[k];
            As[p] = Rkp;
            R[k] = As;
            As = R[k];
            As[q] = Rkq;
            R[k] = As;
        }
        AR[0] = A;
        AR[1] = R;
        return AR;

    }

    //-----------------------------------------------------------------------------------------
    struct MatrixAndVector
    {
        public float3x3 m;
        public float3 v;
    }

    MatrixAndVector eigenDecomposition(float3x3 A)
    {
        const int numJacobiIterations = 10;
        const float epsilon = 1e-15f;

        float3x3 D = A;
        MatrixAndVector eigen = new MatrixAndVector();
        float3x3 eigenVecs;
        float3 eigenVals = math.float3(0);

        // only for symmetric matrices!
        eigenVecs = math.float3x3(1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f);	// unit matrix
        int iter = 0;
        while (iter < numJacobiIterations)  // 3 off diagonal elements
        {
            // find off diagonal element with maximum modulus
            int p, q;
            float a, max;
            max = math.abs(D[0][1]);
            p = 0; q = 1;
            a = math.abs(D[0][2]);
            if (a > max)
            {
                p = 0;
                q = 2;
                max = a;
            }
            a = math.abs(D[1][2]);
            if (a > max)
            {
                p = 1;
                q = 2;
                max = a;
            }
            // all small enough -> done
            if (max < epsilon) break;
            // rotate matrix with respect to that element
            NativeArray<float3x3> j = jacobiRotate(D, eigenVecs, p, q);
            D = j[0];
            eigenVecs = j[1];
            iter++;
            j.Dispose();
        }
        eigenVals[0] = D[0][0];
        eigenVals[1] = D[1][1];
        eigenVals[2] = D[2][2];

        eigen.m = eigenVecs;
        eigen.v = eigenVals;
        return eigen;


    }

    /** Perform polar decomposition A = (U D U^T) R
     */
    float3x3 polarDecomposition(float3x3 A)
    {

        // A = SR, where S is symmetric and R is orthonormal
        // -> S = (A A^T)^(1/2)

        // A = U D U^T R

        float3x3 AAT = math.float3x3(0);
        AAT.c0.x = A[0][0] * A[0][0] + A[0][1] * A[0][1] + A[0][2] * A[0][2];
        AAT.c1.y = A[1][0] * A[1][0] + A[1][1] * A[1][1] + A[1][2] * A[1][2];
        AAT.c2.z = A[2][0] * A[2][0] + A[2][1] * A[2][1] + A[2][2] * A[2][2];

        AAT.c0.y = A[0][0] * A[1][0] + A[0][1] * A[1][1] + A[0][2] * A[1][2];
        AAT.c0.z = A[0][0] * A[2][0] + A[0][1] * A[2][1] + A[0][2] * A[2][2];
        AAT.c1.z = A[1][0] * A[2][0] + A[1][1] * A[2][1] + A[1][2] * A[2][2];

        AAT.c1.x = AAT[0][1];
        AAT.c2.x = AAT[0][2];
        AAT.c2.y = AAT[1][2];

        float3x3 R = math.float3x3(1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f);
        MatrixAndVector decomposition = eigenDecomposition(AAT);
        float3x3 U = decomposition.m;
        float3 eigenVals = decomposition.v;

        float d0 = math.sqrt(eigenVals[0]);
        float d1 = math.sqrt(eigenVals[1]);
        float d2 = math.sqrt(eigenVals[2]);

        const float eps = 1e-15f;

        float l0 = eigenVals[0]; if (l0 <= eps) l0 = 0.0f; else l0 = (1.0f) / d0;
        float l1 = eigenVals[1]; if (l1 <= eps) l1 = 0.0f; else l1 = (1.0f) / d1;
        float l2 = eigenVals[2]; if (l2 <= eps) l2 = 0.0f; else l2 = (1.0f) / d2;

        float3x3 S1 = math.float3x3(0);
        S1.c0.x = l0 * U[0][0] * U[0][0] + l1 * U[0][1] * U[0][1] + l2 * U[0][2] * U[0][2];
        S1.c1.y = l0 * U[1][0] * U[1][0] + l1 * U[1][1] * U[1][1] + l2 * U[1][2] * U[1][2];
        S1.c2.z = l0 * U[2][0] * U[2][0] + l1 * U[2][1] * U[2][1] + l2 * U[2][2] * U[2][2];

        S1.c0.y = l0 * U[0][0] * U[1][0] + l1 * U[0][1] * U[1][1] + l2 * U[0][2] * U[1][2];
        S1.c0.z = l0 * U[0][0] * U[2][0] + l1 * U[0][1] * U[2][1] + l2 * U[0][2] * U[2][2];
        S1.c1.z = l0 * U[1][0] * U[2][0] + l1 * U[1][1] * U[2][1] + l2 * U[1][2] * U[2][2];

        S1.c1.x = S1[0][1];
        S1.c2.x = S1[0][2];
        S1.c2.y = S1[1][2];

        R = math.mul(S1, A);

        //stabilize
        float3 c0, c1, c2;
        c0 = R.c0;
        c1 = R.c1;
        c2 = R.c2;
        if (math.lengthsq(c0) < eps)
            c0 = math.cross(c1, c2);
        else if (math.lengthsq(c1) < eps)
            c1 = math.cross(c2, c0);
        else
            c2 = math.cross(c0, c1);

        R.c0 = c0;
        R.c1 = c1;
        R.c2 = c2;

        //R = A;
        return R;
    }


    /** Return the one norm of the matrix.
     */
    float oneNorm(float3x3 A)
    {
        float sum1 = math.abs(A[0][0]) + math.abs(A[1][0]) + math.abs(A[2][0]);
        float sum2 = math.abs(A[0][1]) + math.abs(A[1][1]) + math.abs(A[2][1]);
        float sum3 = math.abs(A[0][2]) + math.abs(A[1][2]) + math.abs(A[2][2]);
        float maxSum = sum1;
        if (sum2 > maxSum)
            maxSum = sum2;
        if (sum3 > maxSum)
            maxSum = sum3;
        return maxSum;
    }


    /** Return the inf norm of the matrix.
     */
    float infNorm(float3x3 A)
    {
        float sum1 = math.abs(A[0][0]) + math.abs(A[0][1]) + math.abs(A[0][2]);
        float sum2 = math.abs(A[1][0]) + math.abs(A[1][1]) + math.abs(A[1][2]);
        float sum3 = math.abs(A[2][0]) + math.abs(A[2][1]) + math.abs(A[2][2]);
        float maxSum = sum1;
        if (sum2 > maxSum)
            maxSum = sum2;
        if (sum3 > maxSum)
            maxSum = sum3;
        return maxSum;
    }
    /** Perform a polar decomposition of matrix M and return the rotation matrix R. This method handles the degenerated cases.
   */
    float3x3 polarDecompositionStable(float3x3 M, float tolerance)
    {
        float3x3 Mt = math.transpose(M);
        float Mone = oneNorm(M);
        float Minf = infNorm(M);
        float Eone;
        float3x3 MadjTt;
        float3x3 Et = math.float3x3(0);
        do
        {
            MadjTt.c0 = math.cross(M.c1, M.c2);
            MadjTt.c1 = math.cross(M.c2, M.c0);
            MadjTt.c2 = math.cross(M.c0, M.c1);
            MadjTt = math.transpose(MadjTt);

            float det = Mt[0][0] * MadjTt[0][0] + Mt[0][1] * MadjTt[0][1] + Mt[0][2] * MadjTt[0][2];

            if (math.abs(det) < 1e-12f)
            {
                float3 len = math.float3(0);
                int index = -1;
                for (int i = 0; i < 3; i++)
                {
                    len[i] = math.lengthsq(MadjTt[i]);
                    if (len[i] > 1e-12f)
                    {
                        // index of valid cross product
                        // => is also the index of the vector in Mt that must be exchanged
                        index = i;
                        break;
                    }
                }
                if (index == -1)
                    return math.float3x3(1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f);
                else
                {
                    Mt[index] = math.cross(Mt[(index + 1) % 3], Mt[(index + 2) % 3]);
                    MadjTt[(index + 1) % 3] = math.cross(Mt[(index + 2) % 3], Mt[index]);
                    MadjTt[(index + 2) % 3] = math.cross(Mt[index], Mt[(index + 1) % 3]);
                    float3x3 M2 = math.transpose(Mt);
                    Mone = oneNorm(M2);
                    Minf = infNorm(M2);
                    det = Mt[0][0] * MadjTt[0][0] + Mt[0][1] * MadjTt[0][1] + Mt[0][2] * MadjTt[0][2];
                }
            }
            float MadjTone = oneNorm(MadjTt);
            float MadjTinf = infNorm(MadjTt);

            float gamma = math.sqrt(math.sqrt((MadjTone * MadjTinf) / (Mone * Minf)) / math.abs(det));

            float g1 = gamma * 0.5f;
            float g2 = 0.5f / (gamma * det);

            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    float3 intermediate = Et[i];
                    intermediate[j] = Mt[i][j];
                    Et[i] = intermediate;
                    intermediate = Mt[i];
                    intermediate[j] = g1 * Mt[i][j] + g2 * MadjTt[i][j];
                    Mt[i] = intermediate;
                    intermediate = Et[i];
                    intermediate[j] -= Mt[i][j];
                    Et[i] = intermediate;
                }
            }

            Eone = oneNorm(Et);
            Mone = oneNorm(Mt);
            Minf = infNorm(Mt);
        } while (Eone > Mone * tolerance);

        // Q = Mt^T

        return math.transpose(Mt);
    }

    #endregion


    #region Math Quad

    float[,] changeToNine(float3 position)
    {
        float[,] result = new float[9, 1];
        result[0, 0] = position.x;
        result[1, 0] = position.y;
        result[2, 0] = position.z;

        result[3, 0] = position.x * position.x;
        result[4, 0] = position.y * position.y;
        result[5, 0] = position.z * position.z;

        result[6, 0] = position.x * position.y;
        result[7, 0] = position.y * position.z;
        result[8, 0] = position.z * position.x;

        return result;
    }

    float[,] multiplic(float[,] first, float[,] second, int m1, int n1, int m2, int n2)
    {
        float[,] result = new float[m1, n2];
        if (n1 != m2)
        {
            Debug.Log("multiplication invalid");
            return result;
        }
        for (int i = 0; i < m1; i++)
            for (int j = 0; j < n2; j++)
            {
                result[i, j] = 0.0f;
                for (int k = 0; k < n1; k++)
                    result[i, j] += first[i, k] * second[k, j];
            }
        return result;


    }

    float[,] makeSum(float[,] first, float[,] second, int m1, int n1, int m2, int n2)
    {
        float[,] result = new float[m1, n2];
        if (m1 != m2 || n1 != n2)
        {
            Debug.Log("invalid sum");
            return result;
        }
        for (int i = 0; i < m1; i++)
            for (int j = 0; j < n1; j++)
                result[i, j] += first[i, j] + second[i, j];
        return result;
    }

    float[,] tranposeMatrix(float[,] matrix)
    {
        float[,] result = new float[1, 9];
        for (int j = 0; j < 9; j++)
        {
            result[0, j] = matrix[j, 0];
        }
        return result;
    }

    float[,] multiplyByNumber(float[,] matrix, int m1, int n1, float number)
    {
        float[,] result = new float[m1, n1];
        for (int i = 0; i < m1; i++)
            for (int j = 0; j < n1; j++)
                result[i, j] = matrix[i, j] * number;
        return result;
    }


    //Implementar inversa

    struct LU
    {
        public float[,] L;
        public float[,] U;
    }

    LU LUDecomposition(float[,] matrix, int n)
    {
        LU decomposition = new LU();
        decomposition.L = new float[n, n];
        decomposition.U = new float[n, n];
        for (int i = 0; i < n; i++)
        {
            decomposition.U[i, i] = 1.0f;

        }
        decomposition.L[0, 0] = matrix[0, 0];
        for (int j = 1; j < n; j++)
        {
            decomposition.L[j, 0] = matrix[j, 0];

            //error divided by zero
            if (decomposition.L[0, 0] == 0)
                Debug.Log("zero in LU decomposition");

            decomposition.U[0, j] = matrix[0, j] / decomposition.L[0, 0];
        }
        for (int j = 2; j <= n - 1; j++)
        {
            for (int i = j; i <= n; i++)
            {
                decomposition.L[i - 1, j - 1] = matrix[i - 1, j - 1];
                for (int k = 1; k <= j - 1; k++)
                    decomposition.L[i - 1, j - 1] -= decomposition.L[i - 1, k - 1] * decomposition.U[k - 1, j - 1];
            }
            for (int k = j + 1; k <= n; k++)
            {
                decomposition.U[j - 1, k - 1] = matrix[j - 1, k - 1];
                for (int i = 1; i <= j - 1; i++)
                    decomposition.U[j - 1, k - 1] -= decomposition.L[j - 1, i - 1] * decomposition.U[i - 1, k - 1];


                //error divided by zero
                if (decomposition.L[j - 1, j - 1] == 0)
                    Debug.Log("zero in LU decomposition");

                decomposition.U[j - 1, k - 1] = decomposition.U[j - 1, k - 1] / decomposition.L[j - 1, j - 1];
            }
        }
        decomposition.L[n - 1, n - 1] = matrix[n - 1, n - 1];
        for (int k = 1; k <= n - 1; k++)
            decomposition.L[n - 1, n - 1] -= decomposition.L[n - 1, k - 1] * decomposition.U[k - 1, n - 1];


        return decomposition;
    }


    //Forward Substitution
    float[,] columnForL(float[,] L, float[,] x, int n)
    {
        float[,] column = new float[n, 1];
        column[0, 0] = x[0, 0] / L[0, 0];
        for (int i = 1; i < n; i++)
        {
            float result = x[i, 0];
            for (int j = 0; j < i; j++)
                result -= L[i, j] * column[j, 0];
            result = result / L[i, i];
            column[i, 0] = result;
        }
        return column;

    }

    //Backward Substitution
    float[,] columnOfInverse(float[,] U, float[,] x, int n)
    {
        float[,] column = new float[n, 1];
        column[n - 1, 0] = x[n - 1, 0] / U[n - 1, n - 1];
        for (int i = n - 2; i >= 0; i--)
        {
            float result = x[i, 0];
            for (int j = n - 1; j > i; j--)
                result -= U[i, j] * column[j, 0];

            //error divided by zero
            if (U[i, i] == 0)
                Debug.Log("zero in U matrix");

            result = result / U[i, i];
            column[i, 0] = result;
        }
        return column;
    }

    float[,] findInverse(float[,] matrix, int n)
    {
        float[,] inverse = new float[n, n];
        LU decomposition = LUDecomposition(matrix, n);
        for (int j = 0; j < n; j++)
        {
            float[,] column = new float[n, 1];
            column[j, 0] = 1.0f;
            column = columnForL(decomposition.L, column, n);
            column = columnOfInverse(decomposition.U, column, n);
            for (int i = 0; i < n; i++)
                inverse[i, j] = column[i, 0];
        }
        return inverse;
    }

    #endregion


    // Use this for initialization
    void Start()
    {
        //matrix[0, 0] = 7.0f;
        //Debug.Log(matrix[0,0]);

        matrix[0, 0] = 3.0f;
        matrix[0, 1] = -0.1f;
        matrix[0, 2] = -0.2f;
        matrix[1, 0] = 0.1f;
        matrix[1, 1] = 7.0f;
        matrix[1, 2] = -0.3f;
        matrix[2, 0] = 0.3f;
        matrix[2, 1] = -0.2f;
        matrix[2, 2] = 10.0f;


        /*matrix[0, 0] = 1.0f;
        matrix[0, 1] = 2.0f;
        matrix[0, 2] = 3.0f;
        matrix[1, 0] = 4.0f;
        matrix[1, 1] = 5.0f;
        matrix[1, 2] = 6.0f;
        matrix[2, 0] = 7.0f;
        matrix[2, 1] = 8.0f;
        matrix[2, 2] = 9.0f;*/

        //matrix[0, 0] = 4.0f;
        // matrix[0, 1] = 7.0f;
        //matrix[1, 0] = 2.0f;
        // matrix[1, 1] = 6.0f;

        /*LU r = LUDecomposition(matrix, 2);
        Debug.Log("L");
        Debug.Log(r.L[0, 0]);
        Debug.Log(r.L[0, 1]);
       // Debug.Log(r.L[0, 2]);
        Debug.Log(r.L[1, 0]);
        Debug.Log(r.L[1, 1]);
       // Debug.Log(r.L[1, 2]);
        //Debug.Log(r.L[2, 0]);
       // Debug.Log(r.L[2, 1]);
       // Debug.Log(r.L[2, 2]);

        Debug.Log("U");
        Debug.Log(r.U[0, 0]);
        Debug.Log(r.U[0, 1]);
       // Debug.Log(r.U[0, 2]);
        Debug.Log(r.U[1, 0]);
        Debug.Log(r.U[1, 1]);
       // Debug.Log(r.U[1, 2]);
       // Debug.Log(r.U[2, 0]);
       // Debug.Log(r.U[2, 1]);
       // Debug.Log(r.U[2, 2]);*/

        Debug.Log("I");
        float[,] inn = findInverse(matrix, 3);
        Debug.Log(inn[0, 0]);
        Debug.Log(inn[0, 1]);

        Debug.Log(inn[0, 2]);

        Debug.Log(inn[1, 0]);
        Debug.Log(inn[1, 1]);

        Debug.Log(inn[1, 2]);
        Debug.Log(inn[2, 0]);
        Debug.Log(inn[2, 1]);
        Debug.Log(inn[2, 2]);



        //test jacobi
        /*float3x3 A = math.float3x3(5.0f, 7.0f, 3.0f, 7.0f, 0.3f, 4.3f, 3.0f, 4.3f, 9.0f);
        float3x3 R = math.float3x3(1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f);
        Debug.Log("A");
        Debug.Log(A);
        NativeArray<float3x3> n= jacobiRotate(A, R, 1, 2);
        A = n[0];
        R = n[1];
        Debug.Log("A");
        Debug.Log(A);
        n.Dispose();*/

        //test eigendecomp

        /*float3x3 A= math.float3x3(3.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 1.0f, 2.0f, 2.0f);
        //float3x3 A= math.float3x3(1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f);
        //float3x3 A= math.float3x3(-1.0f, 2.0f, 2.0f, 2.0f, 2.0f, -1.0f, 2.0f, -1.0f, 2.0f);
        MatrixAndVector e = eigenDecomposition(A);
        Debug.Log("eVec");
        Debug.Log(e.m);
        Debug.Log("eVal");
        Debug.Log(e.v);
        */

        bool quadratic = false;

        dt = Time.fixedDeltaTime;
        bounds = math.float3(4.75f, 4.75f, 4.75f);

        var point_sampler = GameObject.FindObjectOfType<PointSampler>();
        var samples = point_sampler.points;
        var masses = point_sampler.masses;

        // round #samples down to nearest power of 2 if needed, for job system to be able to split workload
        int po2_amnt = 1; while (po2_amnt <= samples.Count) po2_amnt <<= 1;
        num_particles = po2_amnt >> 1;

        ps = new NativeArray<Particle>(num_particles, Allocator.Persistent);
        //psquadratic= new NativeArray<Particle>(num_particles, Allocator.Persistent);

        // populate our array of particles from the samples given, set their initial state
        for (int i = 0; i < num_particles; ++i)
        {
            float3 sample = samples[i];

            Particle p = new Particle();
            p.x = p.p = math.float3(sample.x, sample.y, sample.z);
            p.v = p.f = math.float3(0);

            // setting masses based on the greyscale value of our image
            p.inv_mass = 1.0f / masses[i];

            ps[i] = p;

            ///////////////////////////////////////////////////// For quad

            //QuadParticle quadvec = new QuadParticle();
            //quadvec.quad = new float[9, 1];
            //quadvec.quad[0,0]=p.

        }
        bool could_init = Init_Body();
        if (!could_init) print("Issue initializing shape");

    }

    private bool Init_Body()
    {
        deltas = new NativeArray<float3>(num_particles, Allocator.Persistent);
        com_sums = new NativeArray<float4>(num_particles, Allocator.Persistent);
        shape_matrices = new NativeArray<float3x3>(num_particles, Allocator.Persistent);

        // calculate initial center of mass
        float3 rest_cm = math.float3(0);
        float wsum = 0.0f;

        for (int i = 0; i < num_particles; i++)
        {
            float wi = 1.0f / (ps[i].inv_mass + eps);
            rest_cm += ps[i].x * wi;
            wsum += wi;
        }
        if (wsum == 0.0) return false;
        rest_cm /= wsum;

        // Calculate inverse rest matrix for use in linear deformation shape matching
        float3x3 A = math.float3x3(0, 0, 0);
        for (int i = 0; i < num_particles; i++)
        {
            float3 qi = ps[i].x - rest_cm;

            // Caching the position differences for later, they'll never change
            deltas[i] = qi;

            // this is forming Aqq, the second term of equation (7) in the paper
            float wi = 1.0f / (ps[i].inv_mass + eps);
            float x2 = wi * qi[0] * qi[0];
            float y2 = wi * qi[1] * qi[1];
            float z2 = wi * qi[2] * qi[2];
            float xy = wi * qi[0] * qi[1];
            float xz = wi * qi[0] * qi[2];
            float yz = wi * qi[1] * qi[2];
            A.c0.x += x2; A.c1.x += xy; A.c2.x += xz;
            A.c0.y += xy; A.c1.y += y2; A.c2.y += yz;
            A.c0.z += xz; A.c1.z += yz; A.c2.z += z2;
        }
        float det = math.determinant(A);
        if (math.abs(det) > eps)
        {
            inv_rest_matrix = math.inverse(A);
            return true;
        }
        return false;
    }

    bool Solve_Shape_Matching()
    {
        JobHandle jh;

        // this stride is used to split the linear summation in both the center of mass, and the calculation of matrix Apq, into parts.
        // these are then calculated in parallel, and finally combined serially
        int stride = ps.Length / division;

        // sum up center of mass in parallel
        var job_sum_center_of_mass = new Job_SumCenterOfMass()
        {
            ps = ps,
            com_sums = com_sums,
            stride = stride
        };

        jh = job_sum_center_of_mass.Schedule(num_particles, division);
        jh.Complete();

        // after the job is complete, we have the results of each individual summation stored at every "stride'th" array entry of com_sums.
        // the CoM is a float2, but we store the total weight each batch used in the z component. finally we divide the final value by this lump sum
        float3 cm = math.float3(0);
        float sum = 0;
        for (int i = 0; i < com_sums.Length; i += stride)
        {
            cm.x += com_sums[i].x;
            cm.y += com_sums[i].y;
            cm.z += com_sums[i].z;
            sum += com_sums[i].w;
        }
        cm /= sum;

        // calculating Apq in batches, same idea as used for CoM calculation
        var job_sum_shape_matrix = new Job_SumShapeMatrix()
        {
            cm = cm,
            shape_matrices = shape_matrices,
            ps = ps,
            deltas = deltas,
            stride = stride
        };

        jh = job_sum_shape_matrix.Schedule(num_particles, division);
        jh.Complete();

        // sum up batches and then normalize by total batch count.
        float3x3 Apq = math.float3x3(0, 0, 0, 0, 0, 0, 0, 0, 0);
        for (int i = 0; i < shape_matrices.Length; i += stride)
        {
            float3x3 shape_mat = shape_matrices[i];
            Apq.c0 += shape_mat.c0;
            Apq.c1 += shape_mat.c1;
            Apq.c2 += shape_mat.c2;
        }
        Apq.c0 /= division;
        Apq.c1 /= division;
        Apq.c2 /= division;


        // Calculate A = Apq * Aqq for linear deformations
        float3x3 A = math.mul(Apq, inv_rest_matrix);

        // calculating the rotation matrix R

        //float3x3 R = polarDecomposition(A);
        const float eps = 1e-6f;
        float3x3 R = polarDecompositionStable(A, eps);

        // volume preservation from Müller paper
        float det_A = math.determinant(A);

        // if our determinant is < 0 here, our shape is inverted. if it's 0, it's collapsed entirely.
        if (det_A != 0)
        {
            // just using the absolute value here for stability in the case of inverted shapes
            float sqrt_det = math.sqrt(math.abs(det_A));
            A.c0 /= sqrt_det;
            A.c1 /= sqrt_det;
            A.c2 /= sqrt_det;
        }

        // blending between simple shape matched rotation (R term) and the area-preserved deformed shape
        // if linear_deformation_blending = 0, we have "standard" shape matching which only supports small changes from the rest shape. try setting this to 1.0f - pretty amazing
        float3x3 A_term = A * linear_deformation_blending;
        float3x3 R_term = R * (1.0f - linear_deformation_blending);

        // "goal position" matrix composed of a linear blend of A and R
        float3x3 GM = math.float3x3(A_term.c0 + R_term.c0, A_term.c1 + R_term.c1, A_term.c2 + R_term.c2);

        // now actually modify particle positions to apply the shape matching
        var j_get_deltas = new Job_GetDeltas()
        {
            cm = cm,
            ps = ps,
            deltas = deltas,
            GM = GM
        };

        jh = j_get_deltas.Schedule(num_particles, division);
        jh.Complete();

        return true;
    }


    // Update is called once per frame
    void FixedUpdate()
    {
        // first integration step applied to all particles
        var j_integrate_0 = new Job_Integrate0()
        {
            ps = ps,
            dt = dt,
            //teste tempo real
            //g = g 
        };

        JobHandle jh = j_integrate_0.Schedule(num_particles, division);
        jh.Complete();

        // starts solving shape matching constraints here.
        // normally you'd solve any other PBD / XPBD constraints here
        // potentially within a loop, to help constraints converge with a desired stiffness
        Solve_Shape_Matching();

        // mouse interaction






        // final integration step, including boundary conditions
        var j_integrate_1 = new Job_Integrate1()
        {
            ps = ps,
            bounds = bounds,
            inv_dt = 1.0f / dt
        };

        jh = j_integrate_1.Schedule(num_particles, division);
        jh.Complete();
    }


    private void OnDestroy()
    {
        ps.Dispose();
        //psquadratic.Dispose();

        deltas.Dispose();
        com_sums.Dispose();
        shape_matrices.Dispose();

    }
}
