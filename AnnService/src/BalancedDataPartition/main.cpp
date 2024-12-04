 // Copyright (c) Microsoft Corporation. All rights reserved.
 // Licensed under the MIT License.

#include <cstdint>
#include <mpi.h>
#include <string>
#include <thread>
#include <cstdlib>
#include <algorithm>
#include<ctime>
#include <unistd.h>
#include "inc/Core/Common.h"
#include "inc/Core/Common/DistanceUtils.h"
#include "inc/Core/Common/Dataset.h"
#include "inc/Core/Common/BKTree.h"
#include "inc/Helper/Logging.h"
#include "inc/Helper/VectorSetReader.h"
#include "inc/Helper/CommonHelper.h"

using namespace SPTAG;
// Macro for I/O Error Checking
#define CHECKIO(ptr, func, bytes, ...) if (ptr->func(bytes, __VA_ARGS__) != bytes) { \
    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "DiskError: Cannot read or write %d bytes.\n", (int)(bytes)); \
    exit(1); \
}

typedef short LabelType;

class PartitionOptions : public Helper::ReaderOptions
{
public:
    PartitionOptions():Helper::ReaderOptions(VectorValueType::Float, 0, VectorFileType::TXT, "|", 32)
    {
        AddRequiredOption(m_inputFiles, "-i", "--input", "Input raw data.");
        AddRequiredOption(m_clusterNum, "-c", "--numclusters", "Number of clusters.");
        AddOptionalOption(m_stopDifference, "-df", "--diff", "Clustering stop center difference.");
        AddOptionalOption(m_maxIter, "-r", "--iters", "Max clustering iterations.");
        AddOptionalOption(m_localSamples, "-s", "--samples", "Number of samples for fast clustering.");
        AddOptionalOption(m_lambda, "-l", "--lambda", "lambda for balanced size level.");
        AddOptionalOption(m_distMethod, "-m", "--dist", "Distance method (L2 or Cosine).");
        AddOptionalOption(m_outdir, "-o", "--outdir", "Output directory.");
        AddOptionalOption(m_weightfile, "-w", "--weight", "vector weight file.");
        AddOptionalOption(m_wlambda, "-lw", "--wlambda", "lambda for balanced weight level.");
        AddOptionalOption(m_seed, "-e", "--seed", "Random seed.");
        AddOptionalOption(m_initIter, "-x", "--init", "Number of iterations for initialization.");
        AddOptionalOption(m_clusterassign, "-a", "--assign", "Number of clusters to be assigned."); // 每个向量最多被分配到几个簇
        AddOptionalOption(m_vectorfactor, "-vf", "--vectorscale", "Max vector number scale factor.");
        AddOptionalOption(m_closurefactor, "-cf", "--closurescale", "Max closure factor"); // 距离比较时的RNG参数
        AddOptionalOption(m_stage, "-g", "--stage", "Running function (Clustering or LocalPartition)");
        AddOptionalOption(m_centers, "-ct", "--centers", "File path to store centers.");
        AddOptionalOption(m_labels, "-lb", "--labels", "File path to store labels.");
        AddOptionalOption(m_status, "-st", "--status", "Cosmos path to store intermediate centers.");
        AddOptionalOption(m_totalparts, "-tp", "--parts", "Total partitions.");
        AddOptionalOption(m_syncscript, "-ss", "--script", "Run sync script.");
        AddOptionalOption(m_recoveriter, "-ri", "--recover", "Recover iteration.");
        AddOptionalOption(m_newp, "-np", "--newpenalty", "old penalty: 0, new penalty: 1"); //惩罚
        AddOptionalOption(m_hardcut, "-hc", "--hard", "soft: 0, hard: 1");
    }

    ~PartitionOptions() {}

    std::string m_inputFiles;
    int m_clusterNum;

    float m_stopDifference = 0.000001f;
    int m_maxIter = 100;
    int m_localSamples = 1000000; // 1000->1000000, 用于初始化聚类中心
    float m_lambda = 0.00000f; // 惩罚项
    float m_wlambda = 0.00000f; // 惩罚项
    int m_seed = -1;
    int m_initIter = 3;
    int m_clusterassign = 1; // 每个向量最多被分配到几个簇
    int m_totalparts = 1;
    int m_recoveriter = -1;
    float m_vectorfactor = 1.0f; // 
    float m_closurefactor = 1.0f; // 1.2->1
    int m_newp = 0;
    int m_hardcut = 0; // 开启限制
    DistCalcMethod m_distMethod = DistCalcMethod::L2;

    std::string m_labels = "labels.bin";
    std::string m_centers = "centers.bin";
    std::string m_outdir = "-";
    std::string m_outfile = "vectors.bin";
    std::string m_outmetafile = "meta.bin";
    std::string m_outmetaindexfile = "metaindex.bin";
    std::string m_weightfile = "-";
    std::string m_stage = "Clustering";
    std::string m_status = ".";
    std::string m_syncscript = "";
} options;

EdgeCompare g_edgeComparer;

template <typename T>
bool LoadCenters(T* centers, SizeType row, DimensionType col, const std::string& centerpath, float* lambda = nullptr, float* diff = nullptr, float* mindist = nullptr, int* noimprovement = nullptr) {
    if (fileexists(centerpath.c_str())) {
        auto ptr = f_createIO();
        if (ptr == nullptr || !ptr->Initialize(centerpath.c_str(), std::ios::binary | std::ios::in)) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read center file %s.\n", centerpath.c_str());
            return false;
        }

        SizeType r;
        DimensionType c;
        float f;
        int i;
        if (ptr->ReadBinary(sizeof(SizeType), (char*)&r) != sizeof(SizeType)) return false;
        if (ptr->ReadBinary(sizeof(DimensionType), (char*)&c) != sizeof(DimensionType)) return false;

        if (r != row || c != col) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Row(%d,%d) or Col(%d,%d) cannot match.\n", r, row, c, col);
            return false;
        }

        if (ptr->ReadBinary(sizeof(T) * row * col, (char*)centers) != sizeof(T) * row * col) return false;

        if (lambda) {
            if (ptr->ReadBinary(sizeof(float), (char*)&f) == sizeof(float)) *lambda = f;
        }
        if (diff) {
            if (ptr->ReadBinary(sizeof(float), (char*)&f) == sizeof(float)) *diff = f;
        }
        if (mindist) {
            if (ptr->ReadBinary(sizeof(float), (char*)&f) == sizeof(float)) *mindist = f;
        }
        if (noimprovement) {
            if (ptr->ReadBinary(sizeof(int), (char*)&i) == sizeof(int)) *noimprovement = i;
        }
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Load centers(%d,%d) from file %s.\n", row, col, centerpath.c_str());
        return true;
    }
    return false;
}

template <typename T>
void SaveCenters(T* centers, SizeType row, DimensionType col, const std::string& centerpath, float lambda = 0.0, float diff = 0.0, float mindist = 0.0, int noimprovement = 0) {
    auto ptr = f_createIO();
    if (ptr == nullptr || !ptr->Initialize(centerpath.c_str(), std::ios::binary | std::ios::out)) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to open center file %s to write.\n", centerpath.c_str());
        exit(1);
    }

    CHECKIO(ptr, WriteBinary, sizeof(SizeType), (char*)&row);
    CHECKIO(ptr, WriteBinary, sizeof(DimensionType), (char*)&col);
    CHECKIO(ptr, WriteBinary, sizeof(T) * row * col, (char*)centers);
    CHECKIO(ptr, WriteBinary, sizeof(float), (char*)&lambda);
    CHECKIO(ptr, WriteBinary, sizeof(float), (char*)&diff);
    CHECKIO(ptr, WriteBinary, sizeof(float), (char*)&mindist);
    CHECKIO(ptr, WriteBinary, sizeof(int), (char*)&noimprovement);
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Save centers(%d,%d) to file %s.\n", row, col, centerpath.c_str());
}

template <typename T>
inline float MultipleClustersAssign(const COMMON::Dataset<T>& data,
    std::vector<SizeType>& indices,
    const SizeType first, const SizeType last, COMMON::KmeansArgs<T>& args, COMMON::Dataset<LabelType>& label, bool updateCenters, float lambda, std::vector<float>& weights, float wlambda) {
    float currDist = 0;
    //每个线程分配到的数据集大小 
    SizeType subsize = (last - first - 1) / args._T + 1;
    // 质心在args.centers里
    std::uint64_t avgCount = 0;
    // 计算平均cluster大小
    for (int k = 0; k < args._K; k++) avgCount += args.counts[k];
    avgCount /= args._K;

  
    std::vector<float> dist_total(args._K * args._T, 0);

    // 多线程处理数据点分配
    auto func = [&](int tid)
    {
        // 计算该线程处理的数据范围
        SizeType istart = first + tid * subsize;
        SizeType iend = min(first + (tid + 1) * subsize, last);
        SizeType* inewCounts = args.newCounts + tid * args._K;
        float* inewWeightedCounts = args.newWeightedCounts + tid * args._K;
        float* inewCenters = args.newCenters + tid * args._K * args._D;
        SizeType* iclusterIdx = args.clusterIdx + tid * args._K;
        float* iclusterDist = args.clusterDist + tid * args._K;
        float* idist_total = dist_total.data() + tid * args._K;
        float idist = 0;
        std::vector<SPTAG::NodeDistPair> centerDist(args._K, SPTAG::NodeDistPair());
        // 遍历每个数据点
        for (SizeType i = istart; i < iend; i++) {
            for (int k = 0; k < args._K; k++) {
                // avgCount为当前每个簇的平均元素数量
                // 如果让penalty为正,则更不容易被分配到
                // lambda * count
                // count 越大,penalty越大
                // penalty越大，dist越大
                // dist越大，越不容易被分配
                // std::cout << "惩罚\n";
                // std::cout << lambda << std::endl;
                float penalty = lambda * (((options.m_newp == 1) && (args.counts[k] < avgCount)) ? avgCount : args.counts[k]) + wlambda * args.weightedCounts[k];
                // std::cout << "惩罚: " << penalty << std::endl;
                float dist = args.fComputeDistance(data[indices[i]], args.centers + k * args._D, args._D) + penalty;
                // if(lambda!=0)std::cout << dist << " " << penalty << std::endl; 
                centerDist[k].node = k;
                centerDist[k].distance = dist; // 存这个质心到这个点的距离
            }
            // 这里排序,centerDist存的是数据点到聚类中心的距离
            std::sort(centerDist.begin(), centerDist.end(), [](const SPTAG::NodeDistPair& a, const SPTAG::NodeDistPair& b) {
                return (a.distance < b.distance) || (a.distance == b.distance && a.node < b.node);
                });
            // 对于每个聚类中心
            // m_closurefactor 大于 1 时，允许数据点被分配到距离稍远但在阈值内的聚类中心。
            // 数据点还可以被分配到距离不超过最近距离的 1.2 倍的其他聚类中心。例如，如果数据点到最近聚类中心的距离是 10，那么到其他聚类中心的距离在 12 以内的也会被考虑。
            // 应该是解决boundry问题的
            for (int k = 0; k < label.C(); k++) {
                if (centerDist[k].distance <= centerDist[0].distance * options.m_closurefactor) {
                    label[i][k] = (LabelType)(centerDist[k].node);
                    inewCounts[centerDist[k].node]++;
                    inewWeightedCounts[centerDist[k].node] += weights[indices[i]];
                    idist += centerDist[k].distance;
                    idist_total[centerDist[k].node] += centerDist[k].distance;

                    if (updateCenters) {
                        const T* v = (const T*)data[indices[i]];
                        float* center = inewCenters + centerDist[k].node * args._D;
                        for (DimensionType j = 0; j < args._D; j++) center[j] += v[j];
                        if (centerDist[k].distance > iclusterDist[centerDist[k].node]) {
                            iclusterDist[centerDist[k].node] = centerDist[k].distance;
                            iclusterIdx[centerDist[k].node] = indices[i];
                        }
                    }
                    else {
                        if (centerDist[k].distance <= iclusterDist[centerDist[k].node]) {
                            iclusterDist[centerDist[k].node] = centerDist[k].distance;
                            iclusterIdx[centerDist[k].node] = indices[i];
                        }
                    }
                }
                else {
                    label[i][k] = (std::numeric_limits<LabelType>::max)();
                }
            }
        }
        SPTAG::COMMON::Utils::atomic_float_add(&currDist, idist);
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < args._T; i++) { threads.emplace_back(func, i); }
    for (auto& thread : threads) { thread.join(); }

    // newcounts这里汇总了
    for (int i = 1; i < args._T; i++) {
        for (int k = 0; k < args._K; k++) {
            args.newCounts[k] += args.newCounts[i*args._K + k];
            args.newWeightedCounts[k] += args.newWeightedCounts[i*args._K + k];
            dist_total[k] += dist_total[i * args._K + k];
        }
    }

    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "start printing dist_total\n");
    for (int k = 0; k < args._K; k++) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "cluster %d: dist_total:%f, count:%d, new_count:%d\n", k, dist_total[k],args.counts[k],args.newCounts[k]);
        
    }

    if (updateCenters) {
        // 1.遍历所有线程(除了线程0)
        for (int i = 1; i < args._T; i++) {
            // args.newCenters = new T[args._T*args._K * args._D];
            // 2.获取当前线程center起始位置
            float* currCenter = args.newCenters + i*args._K*args._D;
            // 3.累加到线程0的centers中
            for (size_t j = 0; j < ((size_t)args._K) * args._D; j++) 
            {
                args.newCenters[j] += currCenter[j];
            }
            // 4.更新每个簇的最大距离和对应的数据点
            for (int k = 0; k < args._K; k++) {
                if (args.clusterIdx[i*args._K + k] != -1 && args.clusterDist[i*args._K + k] > args.clusterDist[k]) {
                    args.clusterDist[k] = args.clusterDist[i*args._K + k];
                    args.clusterIdx[k] = args.clusterIdx[i*args._K + k];
                }
            }
            //最终在RefineCenters函数中会除以点数得到均值
        }
    }
    else {
        for (int i = 1; i < args._T; i++) {
            for (int k = 0; k < args._K; k++) {
                if (args.clusterIdx[i*args._K + k] != -1 && args.clusterDist[i*args._K + k] <= args.clusterDist[k]) {
                    args.clusterDist[k] = args.clusterDist[i*args._K + k];
                    args.clusterIdx[k] = args.clusterIdx[i*args._K + k];
                }
            }
        }
    }
    return currDist;
}

template <typename T>
inline float HardMultipleClustersAssign(const COMMON::Dataset<T>& data,
    std::vector<SizeType>& indices,
    const SizeType first, const SizeType last, COMMON::KmeansArgs<T>& args, COMMON::Dataset<LabelType>& label, SizeType* mylimit, std::vector<float>& weights,
    const int clusternum, const bool fill) {
    // 总距离
    float currDist = 0;
    // 每个线程处理的数据量
    // first:0, last:data.R()
    SizeType subsize = (last - first - 1) / args._T + 1;

    SPTAG::Edge* items = new SPTAG::Edge[last - first];

    // 每个线程的处理函数
    auto func1 = [&](int tid)
    {
        SizeType istart = first + tid * subsize; // 处理起始
        SizeType iend = min(first + (tid + 1) * subsize, last); // 处理结束
        float* iclusterDist = args.clusterDist + tid * args._K;
        std::vector<SPTAG::NodeDistPair> centerDist(args._K, SPTAG::NodeDistPair());
        for (SizeType i = istart; i < iend; i++) { // 遍历每个数据点
            for (int k = 0; k < args._K; k++) { // 遍历每个簇
                // 计算距离
                float dist = args.fComputeDistance(data[indices[i]], args.centers + k * args._D, args._D);
                // 存距离
                centerDist[k].node = k;
                centerDist[k].distance = dist;
            }
            // 排序,距离小的放前面
            std::sort(centerDist.begin(), centerDist.end(), [](const SPTAG::NodeDistPair& a, const SPTAG::NodeDistPair& b) {
                return (a.distance < b.distance) || (a.distance == b.distance && a.node < b.node);
                });
            // clusternum是什么？ --> 每个向量最多分配到的簇的数量-1
            // 检查当前第 clusternum 个最近的聚类中心到数据点的距离是否在一个阈值范围内

            // replicate的要求:
            // 1.距离要满足要求
            if (centerDist[clusternum].distance <= centerDist[0].distance * options.m_closurefactor) {
                // 记录分配到的聚类中心编号
                items[i - first].node = centerDist[clusternum].node;
                // 记录距离
                items[i - first].distance = centerDist[clusternum].distance;
                // 距离原始数据点索引
                items[i - first].tonode = i;
                // 累加这个簇的距离
                iclusterDist[centerDist[clusternum].node] += centerDist[clusternum].distance;
            }
            else {
                items[i - first].node = MaxSize;
                items[i - first].distance = MaxDist;
                items[i - first].tonode = -i-1;
            }
        }
    };

    // 每个线程执行
    {
        std::vector<std::thread> threads;
        for (int i = 0; i < args._T; i++) { threads.emplace_back(func1, i); }
        for (auto& thread : threads) { thread.join(); }
    }

    std::sort(items, items + last - first, g_edgeComparer);

    // 更新每个聚类(簇)的限制值和距离
    // args._T线程数
    for (int i = 0; i< args._T; i++) {
        for (int k = 0; k < args._K; k++) {
            mylimit[k] -= args.newCounts[i * args._K + k];
            if (i > 0) args.clusterDist[k] += args.clusterDist[i * args._K + k];
        }
    }
    std::size_t startIdx = 0;
    // 对每个聚类执行限制处理
    // tonode是原始数据点编号
    for (int i = 0; i < args._K; ++i)
    {
        //找到第一个大于等于i+1的下标
        std::size_t endIdx = std::lower_bound(items, items + last - first, i + 1, g_edgeComparer) - items;
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "cluster %d: avgdist:%f limit:%d, drop:%zu - %zu\n", items[startIdx].node, args.clusterDist[i] / (endIdx - startIdx), mylimit[i], startIdx + mylimit[i], endIdx);
        for (size_t dropID = startIdx + mylimit[i]; dropID < endIdx; ++dropID)
        {
            if (items[dropID].tonode >= 0) items[dropID].tonode = -items[dropID].tonode - 1;
        }
        startIdx = endIdx;
    }


    // c这里就添加到数据集中了
    auto func2 = [&, subsize](int tid)
    {
        // 属于这个线程处理的数据集
        SizeType istart = tid * subsize;
        SizeType iend = min((tid + 1) * subsize, last - first);
        // 第i个线程的newCounts数组
        SizeType* inewCounts = args.newCounts + tid * args._K;
        // 第i个线程的WeightCount数组
        float* inewWeightedCounts = args.newWeightedCounts + tid * args._K;
        float idist = 0;
        for (SizeType i = istart; i < iend; i++) {
            if (items[i].tonode >= 0) {
                // label[i][j] -> 第i个点第j近的簇 
                label[items[i].tonode][clusternum] = (LabelType)(items[i].node);
                // 数量增加
                inewCounts[items[i].node]++;
                inewWeightedCounts[items[i].node] += weights[indices[items[i].tonode]];
                idist += items[i].distance;
            }
            else {
                items[i].tonode = -items[i].tonode - 1;
                label[items[i].tonode][clusternum] = (std::numeric_limits<LabelType>::max)();
            }

            if (fill) {
                for (int k = clusternum + 1; k < label.C(); k++) {
                    label[items[i].tonode][k] = (std::numeric_limits<LabelType>::max)();
                }
            }
        }
        SPTAG::COMMON::Utils::atomic_float_add(&currDist, idist);
    };

    {
        std::vector<std::thread> threads2;
        for (int i = 0; i < args._T; i++) { threads2.emplace_back(func2, i); }
        for (auto& thread : threads2) { thread.join(); }
    }
    delete[] items;

    std::memset(args.counts, 0, sizeof(SizeType) * args._K);
    std::memset(args.weightedCounts, 0, sizeof(float) * args._K);
    // 更新簇的信息
    for (int i = 0; i < args._T; i++) {
        for (int k = 0; k < args._K; k++) {
            args.counts[k] += args.newCounts[i*args._K + k];
            args.weightedCounts[k] += args.newWeightedCounts[i*args._K + k];
        }
    }
    return currDist;
}
// MPI怎么做的?
template <typename T>
void Process(MPI_Datatype type) {
    int rank, size;
    // 初始化mpi环境
    MPI_Init(NULL, NULL);
    // 获取当前进程的rank
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // 获取总的进程数
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "rank: %d size: %d\n", rank, size);
    // std::cout << "rank: " << rank << " size: " << size << std::endl;
    // std::cout << "lambda: " << options.m_lambda << " wlambda: " << options.m_wlambda << std::endl;

    // load vector file
    // 每个进程加载自己的数据文件,文件名中包含进程的rank
    // 1.构造函数
    auto vectorReader = Helper::VectorSetReader::CreateInstance(std::make_shared<Helper::ReaderOptions>(options));
    // 修改为读取子部分数据集
    // options.m_inputFiles = Helper::StrUtils::ReplaceAll(options.m_inputFiles, "u8bin", std::to_string(rank));
    options.m_inputFiles = options.m_inputFiles + "." + std::to_string(rank);
    if (ErrorCode::Success != vectorReader->LoadFile(options.m_inputFiles))
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read input file.\n");
        exit(1);
    }
    // 2.这里实际上才开始读数据
    std::shared_ptr<VectorSet> vectors = vectorReader->GetVectorSet();
    std::shared_ptr<MetadataSet> metas = vectorReader->GetMetadataSet();
     // if(metas.get()==nullptr) std::cout << "no metas\n";
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,"inputfile: %s, vectors: %d, dimension: %d\n", options.m_inputFiles.c_str(), vectors->Count(), vectors->Dimension());
    // normalize vectors
    if (options.m_distMethod == DistCalcMethod::Cosine) vectors->Normalize(options.m_threadNum);

    // 加载权重
    std::vector<float> weights(vectors->Count(), 0.0f);
    if (options.m_weightfile.compare("-") != 0) {
        options.m_weightfile = Helper::StrUtils::ReplaceAll(options.m_weightfile, "*", std::to_string(rank));
        std::ifstream win(options.m_weightfile, std::ifstream::binary);
        if (!win.is_open()) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Rank %d failed to read weight file %s.\n", rank, options.m_weightfile.c_str());
            exit(1);
        }
        SizeType rows;
        win.read((char*)&rows, sizeof(SizeType));
        if (rows != vectors->Count()) {
            win.close();
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Number of weights (%d) is not equal to number of vectors (%d).\n", rows, vectors->Count());
            exit(1);
        }
        win.read((char*)weights.data(), sizeof(float)*rows);
        win.close();
    }

    // counts = new SizeType[_K];
    // newCounts = new SizeType[_T * _K];
    // 数据
    COMMON::Dataset<T> data(vectors->Count(), vectors->Dimension(), 1024*1024, vectors->Count() + 1, (T*)vectors->GetData());
    COMMON::KmeansArgs<T> args(options.m_clusterNum, vectors->Dimension(), vectors->Count(), options.m_threadNum, options.m_distMethod);
    // 行数: 向量的数量
    // 列数: 该向量最多被分配到几个簇
    COMMON::Dataset<LabelType> label(vectors->Count(), options.m_clusterassign, vectors->Count(), vectors->Count());

    std::vector<SizeType> localindices(data.R(), 0);
    for (SizeType i = 0; i < data.R(); i++) localindices[i] = i;

    // 计算所有进程的 localCount 之和，得到 totalCount(这个会分发给所有进程,数值为数据集的总点数)
    // 这里进行一次通信, sum各个节点的数据量
    unsigned long long localCount = data.R(), totalCount;
    MPI_Allreduce(&localCount, &totalCount, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    // 每个簇最多的点的数量 = totalCount / 簇的数量 * vectorfactor(最多复制因子,每个向量最多被分到几个簇)
    totalCount = static_cast<unsigned long long>(totalCount * 1.0 / args._K * options.m_vectorfactor);

    
    // root线程
    if (rank == 0 && options.m_maxIter > 0 && options.m_lambda < -1e-6f) {
        float fBalanceFactor = COMMON::DynamicFactorSelect<T>(data, localindices, 0, data.R(), args, data.R());
        options.m_lambda = COMMON::Utils::GetBase<T>() * COMMON::Utils::GetBase<T>() / fBalanceFactor / data.R();
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "lambda is set to %f\n", options.m_lambda);
    }
    // 把lambda广播给所有进程
    // 每一个你调用的集体通信方法都是同步的
    MPI_Bcast(&(options.m_lambda), 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "rank %d  data:(%d,%d) machines:%d clusters:%d type:%d threads:%d lambda:%f samples:%d maxcountperpartition:%d\n",
        rank, data.R(), data.C(), size, options.m_clusterNum, ((int)options.m_inputValueType), options.m_threadNum, options.m_lambda, options.m_localSamples, totalCount);


    // only for root process.
    if (rank == 0) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "rank 0 init centers\n");
        // 这里load center应该是没有的
        if (!LoadCenters(args.newTCenters, args._K, args._D, options.m_centers, &(options.m_lambda))) {
            // 注意这里随机种子是随机初始化的
            if (options.m_seed >= 0) std::srand(options.m_seed);
            // 随机挑选一些点作为质心(多次迭代挑选最好的)
            COMMON::InitCenters<T, T>(data, localindices, 0, data.R(), args, options.m_localSamples, options.m_initIter);
        }
    }

    float currDiff = 1.0, d, currDist, minClusterDist = MaxDist;
    int iteration = 0;
    int noImprovement = 0;
    //开始迭代更新质心
    while (currDiff > options.m_stopDifference && iteration < options.m_maxIter) {
        if (rank == 0) {
            // 为什么是args.newTCenters,因为在分配时是存储在newTCenters里           
            std::memcpy(args.centers, args.newTCenters, sizeof(T)*args._K*args._D);
        }
        // 广播质心,同步
        // root初始化质心后,把这个质心广播给所有进程 root的数据发给所有其他进程
        MPI_Bcast(args.centers, args._K*args._D, type, 0, MPI_COMM_WORLD);

        args.ClearCenters();
        args.ClearCounts();
        args.ClearDists(-MaxDist);
        // 根据质心分配数据点
        d = MultipleClustersAssign<T>(data, localindices, 0, data.R(), args, label, true, (iteration == 0) ? 0.0f : options.m_lambda, weights, (iteration == 0) ? 0.0f : options.m_wlambda);
        // newCount - counts才是增量
        // newCount表示这次迭代更新后各个簇中向量的总数量
        // std::cout << "newCounts, counts\n";
        // for(int kk = 0 ; kk < args._K ; kk ++){
        //     std::cout << (int32_t)args.newCounts[kk] << " " << (int32_t)args.counts[kk] << std::endl;
        // }
        // 分配完成后,每个节点更新各个簇中向量的数量
        MPI_Allreduce(args.newCounts, args.counts, args._K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        // 更新权重
        MPI_Allreduce(args.newWeightedCounts, args.weightedCounts, args._K, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        // 更新距离---- currDist的具体含义?
        MPI_Allreduce(&d, &currDist, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

        if (currDist < minClusterDist) {
            noImprovement = 0;
            minClusterDist = currDist;
        }
        else {
            noImprovement++;
        }
        if (noImprovement >= 10) break;

        // sum各个质心,更新质心
        if (rank == 0) {
            MPI_Reduce(MPI_IN_PLACE, args.newCenters, args._K * args._D, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
            currDiff = COMMON::RefineCenters<T, T>(data, args);
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "iter %d dist:%f diff:%f\n", iteration, currDist, currDiff);
        } else
            MPI_Reduce(args.newCenters, args.newCenters, args._K * args._D, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

        iteration++;
        // 广播currDiff
        MPI_Bcast(&currDiff, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    // 没进行迭代更新,质心为initcenter得到的
    if (options.m_maxIter == 0) {
        if (rank == 0) {
            std::memcpy(args.centers, args.newTCenters, sizeof(T)*args._K*args._D);
        }
        MPI_Bcast(args.centers, args._K*args._D, type, 0, MPI_COMM_WORLD);
    }
    else {
        if (rank == 0) {
            for (int i = 0; i < args._K; i++)
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "cluster %d contains vectors:%d weights:%f\n", i, args.counts[i], args.weightedCounts[i]);
        }
    }
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,"finish iter cluster\n");


    d = 0;
    for (SizeType i = 0; i < data.R(); i++) localindices[i] = i;
    // 每个簇的最大原属限制
    // data.R是总元素数量
    // size的意义 = 1
    std::vector<SizeType> myLimit(args._K, (options.m_hardcut == 0) ? data.R() : (SizeType)(options.m_hardcut * totalCount / size));
    // std::cout << "------------限制:"<<myLimit[0] <<"------------\n";
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "rank:%d, limit:%d\n", rank, myLimit[0]);
    std::memset(args.counts, 0, sizeof(SizeType)*args._K);
    args.ClearCounts();
    args.ClearDists(0);
    // m_clusterassign:每个向量最多被分到几个簇
    // 第一次迭代,记录最近的聚类
    // 第二次迭代,记录第二的聚类
    for (int i = 0; i < options.m_clusterassign - 1; i++) {
        d += HardMultipleClustersAssign<T>(data, localindices, 0, data.R(), args, label, myLimit.data(), weights, i, false);
        std::memcpy(myLimit.data(), args.counts, sizeof(SizeType)*args._K);
        // 更新counts和weightedCounts
        MPI_Allreduce(MPI_IN_PLACE, args.counts, args._K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, args.weightedCounts, args._K, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        if (rank == 0) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "assign %d....................d:%f\n", i, d);
            for (int i = 0; i < args._K; i++)
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "cluster %d contains vectors:%d weights:%f\n", i, args.counts[i], args.weightedCounts[i]);
        }
        // size表示有多少台机器
        for (int k = 0; k < args._K; k++)
            if (totalCount > args.counts[k]) // ？
                myLimit[k] += (SizeType)((totalCount - args.counts[k]) / size);
    }
    d += HardMultipleClustersAssign<T>(data, localindices, 0, data.R(), args, label, myLimit.data(), weights, options.m_clusterassign - 1, true);
    std::memcpy(args.newCounts, args.counts, sizeof(SizeType)*args._K);
    // 更新counts、weightcount、距离
    MPI_Allreduce(args.newCounts, args.counts, args._K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, args.weightedCounts, args._K, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&d, &currDist, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,"finish hard assign\n");

    // save label
    if (label.Save(options.m_labels + "." + std::to_string(rank)) != ErrorCode::Success) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to save labels.\n");
        exit(1);
    }
    if (rank == 0) {
        SaveCenters(args.centers, args._K, args._D, options.m_centers, options.m_lambda);
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "final dist:%f\n", currDist);
        for (int i = 0; i < args._K; i++)
            SPTAGLIB_LOG(Helper::LogLevel::LL_Status, "cluster %d contains vectors:%d weights:%f\n", i, args.counts[i], args.weightedCounts[i]);
    }
    // 同步点
    MPI_Barrier(MPI_COMM_WORLD);

    //处理每个聚类结果的输出保存阶段
    if (options.m_outdir.compare("-") != 0) {
        for (int i = 0; i < args._K; i++) { // 遍历每个簇
            if (i % size == rank) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Cluster %d start ......\n", i);
            }
            noImprovement = 0;
            std::string vecfile = options.m_outdir + "/" + options.m_outfile + "." + std::to_string(i + 1);
            if (fileexists(vecfile.c_str())) noImprovement = 1;
            MPI_Allreduce(MPI_IN_PLACE, &noImprovement, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            if (noImprovement) continue;
            
            // 这个簇属于这个节点
            if (i % size == rank) {
                // 直接分区好?
                std::string vecfile = options.m_outdir + "/" + options.m_outfile + "." + std::to_string(i);
                std::string metafile = options.m_outdir + "/" + options.m_outmetafile + "." + std::to_string(i);
                std::string metaindexfile = options.m_outdir + "/" + options.m_outmetaindexfile + "." + std::to_string(i);
                std::shared_ptr<Helper::DiskIO> out = f_createIO(), metaout = f_createIO(), metaindexout = f_createIO();
                if (out == nullptr || !out->Initialize(vecfile.c_str(), std::ios::binary | std::ios::out)) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot open %s to write.\n", vecfile.c_str());
                    exit(1);
                }
                if (metaout == nullptr || !metaout->Initialize(metafile.c_str(), std::ios::binary | std::ios::out)) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot open %s to write.\n", metafile.c_str());
                    exit(1);
                }
                if (metaindexout == nullptr || !metaindexout->Initialize(metaindexfile.c_str(), std::ios::binary | std::ios::out)) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot open %s to write.\n", metaindexfile.c_str());
                    exit(1);
                }
                // out为写向量文件, metaout为写meta文件, metaindexout为写meta索引文件
                CHECKIO(out, WriteBinary, sizeof(int), (char*)(&args.counts[i]));
                CHECKIO(out, WriteBinary, sizeof(int), (char*)(&args._D));
                if (metas != nullptr) CHECKIO(metaindexout, WriteBinary, sizeof(int), (char*)(&args.counts[i]));

                std::uint64_t offset = 0;
                T* recvbuf = args.newTCenters;
                int recvmetabuflen = 200;
                char* recvmetabuf = new char [recvmetabuflen];
                for (int j = 0; j < size; j++) { // 从其他节点接收数据
                    uint64_t offset_before = offset;
                    if (j != rank) {
                        int recv = 0;
                        // 阻塞调用,得到这个簇的数量recv
                        MPI_Recv(&recv, 1, MPI_INT, j, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        for (int k = 0; k < recv; k++) {
                            // 应该是接收数据
                            MPI_Recv(recvbuf, args._D, type, j, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                            CHECKIO(out, WriteBinary, sizeof(T)* args._D, (char*)recvbuf);

                            if (metas != nullptr) {
                                int len;
                                MPI_Recv(&len, 1, MPI_INT, j, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                                if (len > recvmetabuflen) {
                                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "enlarge recv meta buf to %d\n", len);
                                    delete[] recvmetabuf;
                                    recvmetabuflen = len;
                                    recvmetabuf = new char[recvmetabuflen];
                                }
                                MPI_Recv(recvmetabuf, len, MPI_CHAR, j, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                                CHECKIO(metaout, WriteBinary, len, recvmetabuf);
                                CHECKIO(metaindexout, WriteBinary, sizeof(std::uint64_t), (char*)(&offset));
                                offset += len;
                            }
                        }
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "rank %d <- rank %d: %d vectors, %llu bytes meta\n", rank, j, recv, (offset - offset_before));
                    }
                    else {
                        size_t total_rec = 0;
                        for (int k = 0; k < data.R(); k++) {
                            for (int kk = 0; kk < label.C(); kk++) {
                                if (label[k][kk] == (LabelType)i) {
                                    CHECKIO(out, WriteBinary, sizeof(T) * args._D, (char*)(data[localindices[k]]));
                                    if (metas != nullptr) {
                                        ByteArray meta = metas->GetMetadata(localindices[k]);
                                        CHECKIO(metaout, WriteBinary, meta.Length(), (const char*)meta.Data());
                                        CHECKIO(metaindexout, WriteBinary, sizeof(std::uint64_t), (char*)(&offset));
                                        offset += meta.Length();
                                    }
                                    total_rec++;
                                }
                            }
                        }
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "rank %d <- rank %d: %d(%d) vectors, %llu bytes meta\n", rank, j, args.newCounts[i], total_rec, (offset - offset_before));
                    }
                }
                delete[] recvmetabuf;
                if (metas != nullptr) CHECKIO(metaindexout, WriteBinary, sizeof(std::uint64_t), (char*)(&offset));
                out->ShutDown();
                metaout->ShutDown();
                metaindexout->ShutDown();
            }
            else {
                // 发送这个簇中向量的数量
                int dest = i % size;
                MPI_Send(&args.newCounts[i], 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
                size_t total_len = 0;
                size_t total_rec = 0;
                for (int j = 0; j < data.R(); j++) {
                    for (int kk = 0; kk < label.C(); kk++) {
                        if (label[j][kk] == (LabelType)i) {
                            // 发送数据
                            MPI_Send(data[localindices[j]], args._D, type, dest, 1, MPI_COMM_WORLD);
                            // 发送元数据
                            if (metas != nullptr) {
                                ByteArray meta = metas->GetMetadata(localindices[j]);
                                int len = (int)meta.Length();
                                MPI_Send(&len, 1, MPI_INT, dest, 2, MPI_COMM_WORLD);
                                MPI_Send(meta.Data(), len, MPI_CHAR, dest, 3, MPI_COMM_WORLD);
                                total_len += len;
                            }
                            total_rec++;
                        }
                    }
                }
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "rank %d -> rank %d: %d(%d) vectors, %llu bytes meta\n", rank, dest, args.newCounts[i], total_rec, total_len);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
    MPI_Finalize();
}

template <typename T>
ErrorCode SyncSaveCenter(COMMON::KmeansArgs<T> &args, int rank, int iteration, unsigned long long localCount, float localDist, float lambda, float diff, float mindist, int noimprovement, int savecenters, bool assign = false)
{
    if (!direxists(options.m_status.c_str())) mkdir(options.m_status.c_str());
    std::string folder = options.m_status + FolderSep + std::to_string(iteration);
    if (!direxists(folder.c_str())) mkdir(folder.c_str());

    if (!direxists(folder.c_str())) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot create the folder %s.\n", folder.c_str());
        exit(1);
    }

    if (rank == 0 && savecenters > 0) {
        SaveCenters(args.newTCenters, args._K, args._D, folder + FolderSep + "centers.bin", lambda, diff, mindist, noimprovement);
    }

    std::string savePath = folder + FolderSep + "status." + std::to_string(iteration) + "." + std::to_string(rank);
    auto out = f_createIO();
    if (out == nullptr || !out->Initialize(savePath.c_str(), std::ios::binary | std::ios::out)) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot open %s to write status.\n", savePath.c_str());
        exit(1);
    }

    CHECKIO(out, WriteBinary, sizeof(unsigned long long), (const char*)&localCount);
    CHECKIO(out, WriteBinary, sizeof(float), (const char*)&localDist);
    CHECKIO(out, WriteBinary, sizeof(float) * args._K * args._D, (const char*)args.newCenters);
    if (assign) {
        CHECKIO(out, WriteBinary, sizeof(int) * args._K, (const char*)args.counts);
        CHECKIO(out, WriteBinary, sizeof(float) * args._K, (const char*)args.weightedCounts);
    }
    else {
        CHECKIO(out, WriteBinary, sizeof(int) * args._K, (const char*)args.newCounts);
        CHECKIO(out, WriteBinary, sizeof(float) * args._K, (const char*)args.newWeightedCounts);
    }
    out->ShutDown();

    if (!options.m_syncscript.empty()) {        
        try {
        	int return_value = system((options.m_syncscript + " upload " + folder + " " + std::to_string(options.m_totalparts) + " " + std::to_string(savecenters)).c_str());
        	if (return_value != 0)
        		throw std::system_error(errno, std::generic_category(), "error executing command");
        }
        catch (const std::system_error& e) {
        	std::cerr << "error executing command: " << options.m_syncscript << e.what() << '\n';
        	return ErrorCode::Fail;
        }
    }
    else {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Error: Sync script is empty.\n");
    }
    return ErrorCode::Success;
}

template <typename T>
ErrorCode SyncLoadCenter(COMMON::KmeansArgs<T>& args, int rank, int iteration, unsigned long long &totalCount, float &currDist, float &lambda, float &diff, float &mindist, int &noimprovement, bool loadcenters)
{
    std::string folder = options.m_status + FolderSep + std::to_string(iteration);

    //TODO download
    if (!options.m_syncscript.empty()) {
        try {
            // system调用执行命令
            // 用于分布式？
            int return_value = system((options.m_syncscript + " download " + folder + " " + std::to_string(options.m_totalparts) + " " + std::to_string(loadcenters)).c_str());
            if (return_value != 0)
                throw std::system_error(errno, std::generic_category(), "error executing command");
        }
        catch (const std::system_error& e) {
            std::cerr << "error executing command: " << options.m_syncscript << e.what() << '\n';
            return ErrorCode::Fail;
        }
    }
    else {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Error: Sync script is empty.\n");
    }

    if (loadcenters) {
        if (!LoadCenters(args.newTCenters, args._K, args._D, folder + FolderSep + "centers.bin", &lambda, &diff, &mindist, &noimprovement)) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot load centers.\n");
            exit(1);
        }
    }

    memset(args.newCenters, 0, sizeof(float) * args._K * args._D);
    memset(args.counts, 0, sizeof(int) * args._K);
    memset(args.weightedCounts, 0, sizeof(float) * args._K);
    std::unique_ptr<char[]> buf(new char[sizeof(float) * args._K * args._D]);
    unsigned long long localCount;
    float localDist;

    totalCount = 0;
    currDist = 0;
    // 遍历所有分区结果
    for (int part = 0; part < options.m_totalparts; part++) {
        // 加载分区状态文件
        std::string loadPath = folder + FolderSep + "status." + std::to_string(iteration) + "." + std::to_string(part);
        auto input = f_createIO();
        if (input == nullptr || !input->Initialize(loadPath.c_str(), std::ios::binary | std::ios::in)) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot open %s to read status.", loadPath.c_str());
            exit(1);
        }

        // 读取并累加:
        // 1.数据量
        CHECKIO(input, ReadBinary, sizeof(unsigned long long), (char*)&localCount);
        totalCount += localCount;

        // 2.距离和
        CHECKIO(input, ReadBinary, sizeof(float), (char*)&localDist);
        currDist += localDist;

        // 3.中心点坐标
        CHECKIO(input, ReadBinary, sizeof(float) * args._K * args._D, buf.get());
        for (int i = 0; i < args._K * args._D; i++) args.newCenters[i] += *((float*)(buf.get()) + i);

        // 4.各簇点数
        CHECKIO(input, ReadBinary, sizeof(int) * args._K, buf.get());
        for (int i = 0; i < args._K; i++) {
            int partsize = *((int*)(buf.get()) + i);
            if (partsize >= 0 && args.counts[i] <= MaxSize - partsize) args.counts[i] += partsize;
            else {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cluster %d counts overflow:%d + %d(%d)! Set it to MaxSize.\n", i, args.counts[i], partsize, part);
                args.counts[i] = MaxSize;
            }
        }
        // 5.权重和
        CHECKIO(input, ReadBinary, sizeof(float) * args._K, buf.get());
        for (int i = 0; i < args._K; i++) args.weightedCounts[i] += *((float*)(buf.get()) + i);
    }
    return ErrorCode::Success;
}

template <typename T>
void ProcessWithoutMPI() {
    // 读取label文件(这个文件目前还不存在,m_labels表示的是希望写入label到哪个文件),读取rank
    // label.bin.0 -> rank 0
    // std::string rankstr = options.m_labels.substr(options.m_labels.rfind(".") + 1);
    // int rank = std::stoi(rankstr);
    // 这里改成默认rank0
    int rank = 0;
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "DEBUG:rank--%d labels--%s\n", rank, options.m_labels.c_str());

    // 创建vectorReader实例
    auto vectorReader = Helper::VectorSetReader::CreateInstance(std::make_shared<Helper::ReaderOptions>(options));
    // 设置数据文件
    options.m_inputFiles = Helper::StrUtils::ReplaceAll(options.m_inputFiles, "*", std::to_string(rank));
    if (ErrorCode::Success != vectorReader->LoadFile(options.m_inputFiles))
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read input file.\n");
        exit(1);
    }
    // 得到raw data
    std::shared_ptr<VectorSet> vectors = vectorReader->GetVectorSet();
    // 得到metadata,估计没有
    std::shared_ptr<MetadataSet> metas = vectorReader->GetMetadataSet();
    if (vectors->Dimension() != options.m_dimension) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "vector dimension %d is not equal to the dimension %d of the option.\n", vectors->Dimension(), options.m_dimension);
        exit(1);
    }
    // cosine的话normalize
    if (options.m_distMethod == DistCalcMethod::Cosine) vectors->Normalize(options.m_threadNum);

    // vectors->Count为向量的数目
    std::vector<float> weights(vectors->Count(), 0.0f);

    // 读取权重文件,默认没有
    if (options.m_weightfile.compare("-") != 0) {
        options.m_weightfile = Helper::StrUtils::ReplaceAll(options.m_weightfile, "*", std::to_string(rank));
        std::ifstream win(options.m_weightfile, std::ifstream::binary);
        if (!win.is_open()) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Rank %d failed to read weight file %s.\n", rank, options.m_weightfile.c_str());
            exit(1);
        }
        SizeType rows;
        win.read((char*)&rows, sizeof(SizeType));
        if (rows != vectors->Count()) {
            win.close();
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Number of weights (%d) is not equal to number of vectors (%d).\n", rows, vectors->Count());
            exit(1);
        }
        win.read((char*)weights.data(), sizeof(float) * rows);
        win.close();
    }
    COMMON::Dataset<T> data(vectors->Count(), vectors->Dimension(), 1024*1024, vectors->Count() + 1, (T*)vectors->GetData());
    COMMON::KmeansArgs<T> args(options.m_clusterNum, vectors->Dimension(), vectors->Count(), options.m_threadNum, options.m_distMethod);
    COMMON::Dataset<LabelType> label(vectors->Count(), options.m_clusterassign, vectors->Count(), vectors->Count());
    std::vector<SizeType> localindices(data.R(), 0);
    for (SizeType i = 0; i < data.R(); i++) { // 都赋值为本身
        localindices[i] = i;
    }
    args.ClearCounts();

    unsigned long long totalCount;
    float currDiff = 1.0, d = 0.0, currDist, minClusterDist = MaxDist;
    /**
    允许从指定迭代次数恢复聚类过程
    通过加载该迭代的聚类状态文件实现恢复
    避免重复计算之前的迭代步骤
    
     */
    int iteration = options.m_recoveriter;
    int noImprovement = 0;

    /**
    if m_recoveriter < 0:
        从头开始聚类
    else:
        1. 加载第m_recoveriter次迭代的状态
        2. 从该点继续后续迭代
     */
    if (rank == 0 && iteration < 0) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "rank 0 init centers\n");
        if (!LoadCenters(args.newTCenters, args._K, args._D, options.m_centers, &(options.m_lambda))) {
            if (options.m_seed >= 0) std::srand(options.m_seed);
            if (options.m_maxIter > 0 && options.m_lambda < -1e-6f) {
                float fBalanceFactor = COMMON::DynamicFactorSelect<T>(data, localindices, 0, data.R(), args, data.R());
                options.m_lambda = COMMON::Utils::GetBase<T>() * COMMON::Utils::GetBase<T>() / fBalanceFactor / data.R();
            }
            // 初始化聚类中心
            COMMON::InitCenters<T, T>(data, localindices, 0, data.R(), args, options.m_localSamples, options.m_initIter);
        }
    }
    /**
    
    如果是第一次迭代或从中断点恢复，先保存当前的聚类中心，然后调用 SyncLoadCenter 加载这些中心，确保后续的聚类分配使用的是最新的中心点
     */

     // m.status是啥?
    if (iteration < 0) {
        iteration = 0;
        SyncSaveCenter(args, rank, iteration, data.R(), d, options.m_lambda, currDiff, minClusterDist, noImprovement, 2);
    }
    else {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "recover from iteration:%d\n", iteration);
    }

    /***
    load `iteration`次迭代保存的center
     */
    SyncLoadCenter(args, rank, iteration, totalCount, currDist, options.m_lambda, currDiff, minClusterDist, noImprovement, true);

    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "rank %d  data:(%d,%d) machines:%d clusters:%d type:%d threads:%d lambda:%f samples:%d maxcountperpartition:%d\n",
        rank, data.R(), data.C(), options.m_totalparts, options.m_clusterNum, ((int)options.m_inputValueType), options.m_threadNum, options.m_lambda, options.m_localSamples, static_cast<unsigned long long>(totalCount * 1.0 / args._K * options.m_vectorfactor));

    // 最多进行多少次聚类m_maxIter
    while (noImprovement < 10 && currDiff > options.m_stopDifference && iteration < options.m_maxIter) {
        std::memcpy(args.centers, args.newTCenters, sizeof(T) * args._K * args._D);

        args.ClearCenters();
        args.ClearCounts();
        args.ClearDists(-MaxDist);
        // 分配数据点
        d = MultipleClustersAssign<T>(data, localindices, 0, data.R(), args, label, true, (iteration == 0) ? 0.0f : options.m_lambda, weights, (iteration == 0) ? 0.0f : options.m_wlambda);

        SyncSaveCenter(args, rank, iteration + 1, data.R(), d, options.m_lambda, currDiff, minClusterDist, noImprovement, 0);
        if (rank == 0) { // 主线程 load -> refine -> save
            SyncLoadCenter(args, rank, iteration + 1, totalCount, currDist, options.m_lambda, currDiff, minClusterDist, noImprovement, false);
            // 更新聚类中心
            currDiff = COMMON::RefineCenters<T, T>(data, args);
            if (currDist < minClusterDist) {
                noImprovement = 0;
                minClusterDist = currDist;
            }
            else {
                noImprovement++;
            }
            // 保存
            SyncSaveCenter(args, rank, iteration + 1, data.R(), d, options.m_lambda, currDiff, minClusterDist, noImprovement, 1);
        }
        else {
            SyncLoadCenter(args, rank, iteration + 1, totalCount, currDist, options.m_lambda, currDiff, minClusterDist, noImprovement, true);
        }
        iteration++;

        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "iter %d dist:%f diff:%f\n", iteration, currDist, currDiff);
    }

    // finish聚类
    if (options.m_maxIter == 0) {
        std::memcpy(args.centers, args.newTCenters, sizeof(T) * args._K * args._D);
    }
    else {
        if (rank == 0) {
            for (int i = 0; i < args._K; i++)
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "cluster %d contains vectors:%d weights:%f\n", i, args.counts[i], args.weightedCounts[i]);
        }
    }
    d = 0;
    // 每个簇中向量最多的数量
    totalCount = static_cast<unsigned long long>(totalCount * 1.0 / args._K * options.m_vectorfactor);
    unsigned long long tmpTotalCount;
    for (SizeType i = 0; i < data.R(); i++) localindices[i] = i;
    // 如果hardcut
    // a. 0,则每个簇中最多向量为数据集总向量
    // b. 1,则每个簇中最多向量为hardcut * totalCount / totalparts
    // mylimit似乎是限制的新增加的点的数量？
    std::vector<SizeType> myLimit(args._K, (options.m_hardcut == 0)? data.R() : (SizeType)(options.m_hardcut * totalCount / options.m_totalparts));
    std::memset(args.counts, 0, sizeof(SizeType) * args._K);
    args.ClearCounts();
    args.ClearDists(0);
    for (int i = 0; i < options.m_clusterassign - 1; i++) {
        // 第一次调用,记录最近的聚类
        // 第二次调用，记录最近的聚类
        d += HardMultipleClustersAssign<T>(data, localindices, 0, data.R(), args, label, myLimit.data(), weights, i, false);
        std::memcpy(myLimit.data(), args.counts, sizeof(SizeType) * args._K);
        SyncSaveCenter(args, rank, 10000 + iteration + 1 + i, data.R(), d, options.m_lambda, currDiff, minClusterDist, noImprovement, 0, true);
        SyncLoadCenter(args, rank, 10000 + iteration + 1 + i, tmpTotalCount, currDist, options.m_lambda, currDiff, minClusterDist, noImprovement, false);
        if (rank == 0) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "assign %d....................d:%f\n", i, d);
            for (int i = 0; i < args._K; i++)
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "cluster %d contains vectors:%d weights:%f\n", i, args.counts[i], args.weightedCounts[i]);
        }
        for (int k = 0; k < args._K; k++)
            if (totalCount > args.counts[k])
                myLimit[k] += (SizeType)((totalCount - args.counts[k]) / options.m_totalparts);
    }
    // 最后一次聚类
    d += HardMultipleClustersAssign<T>(data, localindices, 0, data.R(), args, label, myLimit.data(), weights, options.m_clusterassign - 1, true);
    std::memcpy(args.newCounts, args.counts, sizeof(SizeType) * args._K);
    SyncSaveCenter(args, rank, 10000 + iteration + options.m_clusterassign, data.R(), d, options.m_lambda, currDiff, minClusterDist, noImprovement, 0, true);
    SyncLoadCenter(args, rank, 10000 + iteration + options.m_clusterassign, tmpTotalCount, currDist, options.m_lambda, currDiff, minClusterDist, noImprovement, false);

    if (label.Save(options.m_labels) != ErrorCode::Success) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to save labels.\n");
        exit(1);
    }
    if (rank == 0) {

        // 保存质心向量
        SaveCenters(args.centers, args._K, args._D, options.m_centers, options.m_lambda);
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "final dist:%f\n", currDist);
        for (int i = 0; i < args._K; i++)
            SPTAGLIB_LOG(Helper::LogLevel::LL_Status, "cluster %d contains vectors:%d weights:%f\n", i, args.counts[i], args.weightedCounts[i]);
    }
}

// Partition根据标签划分数据
template <typename T>
void Partition() {
    // 要有输出目录
    if (options.m_outdir.compare("-") == 0) return;

    // 解析参数用的
    auto vectorReader = Helper::VectorSetReader::CreateInstance(std::make_shared<Helper::ReaderOptions>(options));
    if (ErrorCode::Success != vectorReader->LoadFile(options.m_inputFiles))
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read input file.\n");
        exit(1);
    }
    std::shared_ptr<VectorSet> vectors = vectorReader->GetVectorSet();
    std::shared_ptr<MetadataSet> metas = vectorReader->GetMetadataSet();
    if (options.m_distMethod == DistCalcMethod::Cosine) vectors->Normalize(options.m_threadNum);

    COMMON::Dataset<T> data(vectors->Count(), vectors->Dimension(), 1024*1024, vectors->Count() + 1, (T*)vectors->GetData());

    // short
    COMMON::Dataset<LabelType> label;
    if (label.Load(options.m_labels, vectors->Count(), vectors->Count()) != ErrorCode::Success) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read labels.\n");
        exit(1);
    }

    std::string taskId = options.m_labels.substr(options.m_labels.rfind(".") + 1);
    for (int i = 0; i < options.m_clusterNum; i++) {
        std::string vecfile = options.m_outdir + "/" + options.m_outfile + "." + taskId + "." + std::to_string(i);
        std::string metafile = options.m_outdir + "/" + options.m_outmetafile + "." + taskId + "." + std::to_string(i);
        std::string metaindexfile = options.m_outdir + "/" + options.m_outmetaindexfile + "." + taskId + "." + std::to_string(i);
        std::shared_ptr<Helper::DiskIO> out = f_createIO(), metaout = f_createIO(), metaindexout = f_createIO();
        if (out == nullptr || !out->Initialize(vecfile.c_str(), std::ios::binary | std::ios::out)) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot open %s to write.\n", vecfile.c_str());
            exit(1);
        }
        if (metaout == nullptr || !metaout->Initialize(metafile.c_str(), std::ios::binary | std::ios::out)) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot open %s to write.\n", metafile.c_str());
            exit(1);
        }
        if (metaindexout == nullptr || !metaindexout->Initialize(metaindexfile.c_str(), std::ios::binary | std::ios::out)) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot open %s to write.\n", metaindexfile.c_str());
            exit(1);
        }

        int rows = data.R(), cols = data.C();
        CHECKIO(out, WriteBinary, sizeof(int), (char*)(&rows));
        CHECKIO(out, WriteBinary, sizeof(int), (char*)(&cols));
        if (metas != nullptr) CHECKIO(metaindexout, WriteBinary, sizeof(int), (char*)(&rows));

        std::uint64_t offset = 0;
        int records = 0;
        for (int k = 0; k < data.R(); k++) {
            for (int kk = 0; kk < label.C(); kk++) {
                if (label[k][kk] == (LabelType)i) {
                    CHECKIO(out, WriteBinary, sizeof(T) * cols, (char*)(data[k]));
                    if (metas != nullptr) {
                        ByteArray meta = metas->GetMetadata(k);
                        CHECKIO(metaout, WriteBinary, meta.Length(), (const char*)meta.Data());
                        CHECKIO(metaindexout, WriteBinary, sizeof(std::uint64_t), (char*)(&offset));
                        offset += meta.Length();
                    }
                    records++;
                }
            }
        }
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "part %s cluster %d: %d vectors, %llu bytes meta.\n", taskId.c_str(), i, records, offset);

        if (metas != nullptr) CHECKIO(metaindexout, WriteBinary, sizeof(std::uint64_t), (char*)(&offset));
        CHECKIO(out, WriteBinary, sizeof(int), (char*)(&records), 0);
        CHECKIO(metaindexout, WriteBinary, sizeof(int), (char*)(&records), 0);

        out->ShutDown();
        metaout->ShutDown();
        metaindexout->ShutDown();
    }
}
void ShowPeakMemoryStatus()
{

    std::ifstream statusFile("/proc/self/status");
    std::string line;
    long vmhwm_kb = -1; // 初始化为 -1 表示未找到

    // 按行读取文件内容
    while (std::getline(statusFile, line)) {
        // 查找以 "VmHWM:" 开头的行
        if (line.compare(0, 6, "VmHWM:") == 0) {
            std::istringstream iss(line);
            std::string key;
            long value;
            std::string unit;

            // 解析行内容
            if (!(iss >> key >> value >> unit)) {
                std::cerr << "解析 VmHWM 行失败。" << std::endl;
                break;
            }

            // 通常单位为 kB
            if (unit == "kB") {
                vmhwm_kb = value;
            } else if (unit == "mB") { // 处理可能的其他单位
                vmhwm_kb = value * 1024;
            } else if (unit == "gB") {
                vmhwm_kb = value * 1024 * 1024;
            } else {
                std::cerr << "未识别的内存单位 '" << unit << "'。假设为 kB。" << std::endl;
                vmhwm_kb = value;
            }
            break; // 找到 VmHWM 后退出循环
        }
    }

    statusFile.close();
    // 转换为 GB
    double vmhwm_gb = static_cast<double>(vmhwm_kb) / 1024 / 1024;

    std::cout << "峰值内存使用量 (VmHWM): " << vmhwm_gb << " GB" << std::endl;
}
int main(int argc, char* argv[]) {
    if (!options.Parse(argc - 1, argv + 1))
    {
        exit(1);
    }
    // 3个阶段选择: 1. Clustering 2.ClusteringWithoutMPI 3.LocalPartition
    if (options.m_stage.compare("Clustering") == 0) {
        switch (options.m_inputValueType) {
        case SPTAG::VectorValueType::Float:
            Process<float>(MPI_FLOAT);
            break;
        case SPTAG::VectorValueType::Int16:
            Process<std::int16_t>(MPI_SHORT);
            break;
        case SPTAG::VectorValueType::Int8:
            Process<std::int8_t>(MPI_CHAR);
            break;
        case SPTAG::VectorValueType::UInt8:
            Process<std::uint8_t>(MPI_CHAR);
            break;
        default:
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Error data type!\n");
        }
    }
    else if (options.m_stage.compare("ClusteringWithoutMPI") == 0) {
        switch (options.m_inputValueType) {
        case SPTAG::VectorValueType::Float:
            ProcessWithoutMPI<float>();
            break;
        case SPTAG::VectorValueType::Int16:
            ProcessWithoutMPI<std::int16_t>();
            break;
        case SPTAG::VectorValueType::Int8:
            ProcessWithoutMPI<std::int8_t>();
            break;
        case SPTAG::VectorValueType::UInt8:
            ProcessWithoutMPI<std::uint8_t>();
            break;
        default:
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Error data type!\n");
        }
    }
    else if (options.m_stage.compare("LocalPartition") == 0) {
        switch (options.m_inputValueType) {
        case SPTAG::VectorValueType::Float:
            Partition<float>();
            break;
        case SPTAG::VectorValueType::Int16:
            Partition<std::int16_t>();
            break;
        case SPTAG::VectorValueType::Int8:
            Partition<std::int8_t>();
            break;
        case SPTAG::VectorValueType::UInt8:
            Partition<std::uint8_t>();
            break;
        default:
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Error data type!\n");
        }
    }
    ShowPeakMemoryStatus();
    return 0;
}
