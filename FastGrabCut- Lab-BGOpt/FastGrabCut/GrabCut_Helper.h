#pragma once

#include "stdafx.h"



/*
 GMM - Gaussian Mixture Model			//高斯分布，又名 正态分布
*/
class GMM			
{
public:
    static const int componentsCount = 5;				//总共5个模型（都服从正态分布，也就有5个均值和方差）

	//Mat=矩阵  Vec=垂直向量  Vec3d=3*1的double向量
	GMM();
    GMM( Mat& _model );
    double operator()( const Vec3d color ) const;					//计算像素属于这个GMM混合高斯模型的概率
    double operator()( int ci, const Vec3d color ) const;			//计算像素属于这个GMM混合高斯模型的第ci个模型的概率 
    int whichComponent( const Vec3d color ) const;					//返回这个像素最有可能属于GMM中的哪个高斯模型（概率最大的那个）	

    void initLearning();
    void addSample( int ci, const Vec3d color );
    void endLearning();

private:
    void calcInverseCovAndDeterm( int ci );	//计算第ci个模型的    协方差的逆矩阵=Inverse covariance     行列式=Determinant
    Mat model;
    double* coefs;	//权值变量起始指针    每一个高斯模型在混合模型中的占比/权值  //模型样本数/所有模型样本数
    double* mean;	//均值变量起始指针
    double* cov;	//协方差起始指针

    double inverseCovs[componentsCount][3][3];	//协方差的逆矩阵
    double covDeterms[componentsCount];			//协方差的行列式的值

    double sums[componentsCount][3];		//像素的RGB三个通道的总和sums（用来计算均值）
    double prods[componentsCount][3][3];	//像素的RGB三个通道的prods（用来计算协方差）
    int sampleCounts[componentsCount];		//每个模型对应的样本像素数量
    int totalSampleCount;
};







template <class TWeight> class GCGraph
{
public:
    GCGraph();
    GCGraph( unsigned int vtxCount, unsigned int edgeCount );
    ~GCGraph();
    void create( unsigned int vtxCount, unsigned int edgeCount );
    int addVtx();				//添加一个点
    void addEdges( int i, int j, TWeight w/*i-j权重*/, TWeight revw/*j->i权重*/ );	//添加边（权重）
    void addTermWeights( int i, TWeight sourceW, TWeight sinkW );	//添加特殊边（权重）
    TWeight maxFlow();
    bool inSourceSegment( int i );		//判断某点是否属于Source这边
private:
    class Vtx
    {
    public:
        Vtx *next; // initialized and used in maxFlow() only
        int parent;
        int first;	//第一个延生出去的边（以此点为起点的边）
        int ts;
        int dist;
        TWeight weight;		//  >0 偏向Source   <0 偏向Sink
        uchar t;	//=0 属于source,, !=0 属于sink
    };
    class Edge
    {
    public:
        int dst;		//边的终点（边的起点未保存，因为程序中是通过点找边，无须保存起点）
        int next;		//下一个兄弟边（起点一样的边）
        TWeight weight;
    };

    std::vector<Vtx> vtcs;		//所有点集合
    std::vector<Edge> edges;	//所有边集合
    TWeight flow;				//保存最大流值
};

template <class TWeight>
GCGraph<TWeight>::GCGraph()
{
    flow = 0;
}
template <class TWeight>
GCGraph<TWeight>::GCGraph( unsigned int vtxCount, unsigned int edgeCount )
{
    create( vtxCount, edgeCount );
}
template <class TWeight>
GCGraph<TWeight>::~GCGraph()
{
}
template <class TWeight>
void GCGraph<TWeight>::create( unsigned int vtxCount, unsigned int edgeCount )
{
    vtcs.reserve( vtxCount );
    edges.reserve( edgeCount + 2 );
    flow = 0;
}

template <class TWeight>
int GCGraph<TWeight>::addVtx()
{
    Vtx v;
    memset( &v, 0, sizeof(Vtx));
    vtcs.push_back(v);
    return (int)vtcs.size() - 1;
}

template <class TWeight>
void GCGraph<TWeight>::addEdges( int i, int j, TWeight w, TWeight revw )
{
    CV_Assert( i>=0 && i<(int)vtcs.size() );
    CV_Assert( j>=0 && j<(int)vtcs.size() );
    CV_Assert( w>=0 && revw>=0 );
    CV_Assert( i != j );

    if( !edges.size() )
        edges.resize( 2 );

    Edge fromI, toI;
    fromI.dst = j;
    fromI.next = vtcs[i].first;
    fromI.weight = w;
    vtcs[i].first = (int)edges.size();
    edges.push_back( fromI );

    toI.dst = i;
    toI.next = vtcs[j].first;
    toI.weight = revw;
    vtcs[j].first = (int)edges.size();
    edges.push_back( toI );
}

template <class TWeight>
void GCGraph<TWeight>::addTermWeights( int i, TWeight sourceW, TWeight sinkW )
{
    CV_Assert( i>=0 && i<(int)vtcs.size() );

    TWeight dw = vtcs[i].weight;
    if( dw > 0 )
        sourceW += dw;
    else
        sinkW -= dw;
    flow += (sourceW < sinkW) ? sourceW : sinkW;
    vtcs[i].weight = sourceW - sinkW;
}

template <class TWeight>
TWeight GCGraph<TWeight>::maxFlow()
{
    const int TERMINAL = -1, ORPHAN = -2;
    Vtx stub, *nilNode = &stub, *first = nilNode, *last = nilNode;
    int curr_ts = 0;
    stub.next = nilNode;
    Vtx *vtxPtr = &vtcs[0];
    Edge *edgePtr = &edges[0];

    std::vector<Vtx*> orphans;

    // initialize the active queue and the graph vertices
    for( int i = 0; i < (int)vtcs.size(); i++ )
    {
        Vtx* v = vtxPtr + i;
        v->ts = 0;
        if( v->weight != 0 )
        {
            last = last->next = v;
            v->dist = 1;
            v->parent = TERMINAL;
            v->t = v->weight < 0;
        }
        else
            v->parent = 0;
    }
    first = first->next;
    last->next = nilNode;
    nilNode->next = 0;

    // run the search-path -> augment-graph -> restore-trees loop
    for(;;)
    {
        Vtx* v, *u;
        int e0 = -1, ei = 0, ej = 0;
        TWeight minWeight, weight;
        uchar vt;

        // grow S & T search trees, find an edge connecting them
        while( first != nilNode )
        {
            v = first;
            if( v->parent )
            {
                vt = v->t;
                for( ei = v->first; ei != 0; ei = edgePtr[ei].next )
                {
                    if( edgePtr[ei^vt].weight == 0 )
                        continue;
                    u = vtxPtr+edgePtr[ei].dst;
                    if( !u->parent )
                    {
                        u->t = vt;
                        u->parent = ei ^ 1;
                        u->ts = v->ts;
                        u->dist = v->dist + 1;
                        if( !u->next )
                        {
                            u->next = nilNode;
                            last = last->next = u;
                        }
                        continue;
                    }

                    if( u->t != vt )
                    {
                        e0 = ei ^ vt;
                        break;
                    }

                    if( u->dist > v->dist+1 && u->ts <= v->ts )
                    {
                        // reassign the parent
                        u->parent = ei ^ 1;
                        u->ts = v->ts;
                        u->dist = v->dist + 1;
                    }
                }
                if( e0 > 0 )
                    break;
            }
            // exclude the vertex from the active list
            first = first->next;
            v->next = 0;
        }

        if( e0 <= 0 )
            break;

        // find the minimum edge weight along the path
        minWeight = edgePtr[e0].weight;
        assert( minWeight > 0 );
        // k = 1: source tree, k = 0: destination tree
        for( int k = 1; k >= 0; k-- )
        {
            for( v = vtxPtr+edgePtr[e0^k].dst;; v = vtxPtr+edgePtr[ei].dst )
            {
                if( (ei = v->parent) < 0 )
                    break;
                weight = edgePtr[ei^k].weight;
                minWeight = MIN(minWeight, weight);
                assert( minWeight > 0 );
            }
            weight = fabs(v->weight);
            minWeight = MIN(minWeight, weight);
            assert( minWeight > 0 );
        }

        // modify weights of the edges along the path and collect orphans
        edgePtr[e0].weight -= minWeight;
        edgePtr[e0^1].weight += minWeight;
        flow += minWeight;

        // k = 1: source tree, k = 0: destination tree
        for( int k = 1; k >= 0; k-- )
        {
            for( v = vtxPtr+edgePtr[e0^k].dst;; v = vtxPtr+edgePtr[ei].dst )
            {
                if( (ei = v->parent) < 0 )
                    break;
                edgePtr[ei^(k^1)].weight += minWeight;
                if( (edgePtr[ei^k].weight -= minWeight) == 0 )
                {
                    orphans.push_back(v);
                    v->parent = ORPHAN;
                }
            }

            v->weight = v->weight + minWeight*(1-k*2);
            if( v->weight == 0 )
            {
               orphans.push_back(v);
               v->parent = ORPHAN;
            }
        }

        // restore the search trees by finding new parents for the orphans
        curr_ts++;
        while( !orphans.empty() )
        {
            Vtx* v2 = orphans.back();
            orphans.pop_back();

            int d, minDist = INT_MAX;
            e0 = 0;
            vt = v2->t;

            for( ei = v2->first; ei != 0; ei = edgePtr[ei].next )
            {
                if( edgePtr[ei^(vt^1)].weight == 0 )
                    continue;
                u = vtxPtr+edgePtr[ei].dst;
                if( u->t != vt || u->parent == 0 )
                    continue;
                // compute the distance to the tree root
                for( d = 0;; )
                {
                    if( u->ts == curr_ts )
                    {
                        d += u->dist;
                        break;
                    }
                    ej = u->parent;
                    d++;
                    if( ej < 0 )
                    {
                        if( ej == ORPHAN )
                            d = INT_MAX-1;
                        else
                        {
                            u->ts = curr_ts;
                            u->dist = 1;
                        }
                        break;
                    }
                    u = vtxPtr+edgePtr[ej].dst;
                }

                // update the distance
                if( ++d < INT_MAX )
                {
                    if( d < minDist )
                    {
                        minDist = d;
                        e0 = ei;
                    }
                    for( u = vtxPtr+edgePtr[ei].dst; u->ts != curr_ts; u = vtxPtr+edgePtr[u->parent].dst )
                    {
                        u->ts = curr_ts;
                        u->dist = --d;
                    }
                }
            }

            if( (v2->parent = e0) > 0 )
            {
                v2->ts = curr_ts;
                v2->dist = minDist;
                continue;
            }

            /* no parent is found */
            v2->ts = 0;
            for( ei = v2->first; ei != 0; ei = edgePtr[ei].next )
            {
                u = vtxPtr+edgePtr[ei].dst;
                ej = u->parent;
                if( u->t != vt || !ej )
                    continue;
                if( edgePtr[ei^(vt^1)].weight && !u->next )
                {
                    u->next = nilNode;
                    last = last->next = u;
                }
                if( ej > 0 && vtxPtr+edgePtr[ej].dst == v2 )
                {
                    orphans.push_back(u);
                    u->parent = ORPHAN;
                }
            }
        }
    }
    return flow;
}

template <class TWeight>
bool GCGraph<TWeight>::inSourceSegment( int i )
{
    CV_Assert( i>=0 && i<(int)vtcs.size() );
    return vtcs[i].t == 0;
};

