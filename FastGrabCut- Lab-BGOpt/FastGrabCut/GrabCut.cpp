#include "GrabCut.h"

Vec3d SuperpixelGrabCut::Vec6d_to_Vec3d(Vec6d v6)
{
	Vec3d v3;
	v3[0] = v6[0];
	v3[1] = v6[1];
	v3[2] = v6[2];
	return v3;
}
Vec2d SuperpixelGrabCut::Vec6d_to_Vec2d(Vec6d v6)
{
	Vec2d v2;
	v2[0] = v6[4];
	v2[1] = v6[5];
	return v2;
}

//计算beta，也就是Gibbs能量项中的第二项（平滑项）中的指数项的beta，用来调整
//高或者低对比度时，两个邻域像素的差别的影响的，例如在低对比度时，两个邻域
//像素的差别可能就会比较小，这时候需要乘以一个较大的beta来放大这个差别，
//在高对比度时，则需要缩小本身就比较大的差别。
//所以我们需要分析整幅图像的对比度来确定参数beta，具体的见论文公式（5）。
/*
  Calculate beta - parameter of GrabCut algorithm.
  beta = 1/(2*avg(sqr(||color[i] - color[j]||)))
*/

double SuperpixelGrabCut::calcBeta( const Mat& img )
{
    double beta = 0;
    for( int y = 0; y < img.rows; y++ )
    {
        for( int x = 0; x < img.cols; x++ )
        {
			//累积所有相邻像素差的点乘
			//计算四个方向邻域两像素的差别，也就是欧式距离或者说二阶范数   只计算图像八领域的一半（left upleft up upright），防止重复计算
			//（当所有像素都算完后，就相当于计算八邻域的像素差了）
            Vec3d color = img.at<Vec3b>(y,x);
            if( x>0 ) // color-left  >0的判断是为了避免在图像边界的时候还计算，导致越界
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y,x-1);
                beta += diff.dot(diff);  //矩阵的点乘，也就是各个元素平方的和
            }
            if( y>0 && x>0 ) // color-upleft
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x-1);
                beta += diff.dot(diff);
            }
            if( y>0 ) // color-up
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x);
                beta += diff.dot(diff);
            }
            if( y>0 && x<img.cols-1) // color-upright
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x+1);
                beta += diff.dot(diff);
            }
        }
    }
    if( beta <= std::numeric_limits<double>::epsilon() )
        beta = 0;
    else
        beta = 1.f / (2 * beta/(4*img.cols*img.rows - 3*img.cols - 3*img.rows + 2) ); //论文公式（5）   beta= 1.0f / (2 * beta/(之前累积点乘的相邻像素的边数));   无向图8领域

    return beta;
}

//计算图每个非端点顶点（也就是每个像素作为图的一个顶点，不包括源点s和汇点t）与邻域顶点
//的边的权值。由于是无向图，我们计算的是八邻域，那么对于一个顶点，我们计算四个方向就行，
//在其他的顶点计算的时候，会把剩余那四个方向的权值计算出来。这样整个图算完后，每个顶点
//与八邻域的顶点的边的权值就都计算出来了。
//这个相当于计算Gibbs能量的第二个能量项（平滑项），具体见论文中公式（4）  论文中公式(11)
/*
  Calculate weights of noterminal vertices of graph.
  beta and gamma - parameters of GrabCut algorithm.
 */

void SuperpixelGrabCut::calcNWeights(vector<Vec6d>& superpixels, vector<vector<int> >& superarcs, vector<vector<double> >& spNWeights, int& sideNum, double beta, double gamma)
{
    //gammaDivSqrt2相当于公式（4）中的gamma * dis(i,j)^(-1)，那么可以知道，
	//当i和j是垂直或者水平关系时，dis(i,j)=1，当是对角关系时，dis(i,j)=sqrt(2.0f)。
	//具体计算时，看下面就明白了
	spNWeights.resize(superpixels.size());

	for(int i = 0; i < superarcs.size(); i++)
	{
		spNWeights[i].resize(superarcs[i].size(), 0);
		for(int j = 0; j < superarcs[i].size(); j++)
		{

			//if(i >= superarcs[i][j]) continue;
			int start = i;
			int end = superarcs[i][j];

			Vec3d diff = Vec6d_to_Vec3d(superpixels[start]) - Vec6d_to_Vec3d(superpixels[end]);
			Vec2d dist2d = Vec6d_to_Vec2d(superpixels[start]) - Vec6d_to_Vec2d(superpixels[end]);
			int spSizeStart = superpixels[start][3];	//超像素尺寸（超像素包含的像素个数）
			int spSizeEnd = superpixels[end][3];
			double sizeRate = (double)spSizeEnd / spSizeStart;	//超像素尺寸比率
			double dist = std::sqrt(dist2d.dot(dist2d));

			spNWeights[i][j] = exp(-beta*diff.dot(diff)) * (gamma * sizeRate / dist);	//i号超像素到其第j个邻居(邻居编号由superarcs[i][j]得到)的权值

		}
	}

}

//检查mask的正确性。mask为通过用户交互或者程序设定的，它是和图像大小一样的单通道灰度图，
//每个像素只能取GC_BGD or GC_FGD or GC_PR_BGD or GC_PR_FGD 四种枚举值，分别表示该像素
//（用户或者程序指定）属于背景、前景、可能为背景或者可能为前景像素。具体的参考：
//ICCV2001“Interactive Graph Cuts for Optimal Boundary & Region Segmentation of Objects in N-D Images”
//Yuri Y. Boykov Marie-Pierre Jolly 
/*
  Check size, type and element values of mask matrix.
 */
void SuperpixelGrabCut::checkMask(const vector<Vec6d>& superpixels, const vector<int>& spMask )
{
	if(spMask.size() != superpixels.size())
		CV_Error( CV_StsBadArg, "mask must have as many rows and cols as img" );

	for(int i = 0; i < spMask.size(); i++)
	{
		if(spMask[i] != GC_BGD && spMask[i] != GC_FGD && spMask[i] != GC_PR_BGD && spMask[i] != GC_PR_FGD)
			CV_Error( CV_StsBadArg, "mask element value must be equel"
                    "GC_BGD or GC_FGD or GC_PR_BGD or GC_PR_FGD" );
	}

}

//通过用户框选目标rect来创建mask，rect外的全部作为背景，设置为GC_BGD，
//rect内的设置为 GC_PR_FGD（可能为前景）
/*
  Initialize mask using rectangular.
*/
void SuperpixelGrabCut::initMaskWithRect(vector<int>& klabels, int width, vector<int>& mask, int size, Rect rect )
{
	mask.resize(size, GC_BGD);
	for(int r = 0; r < rect.height; r++)
	{
		for(int c = 0; c < rect.width; c++)
		{
			int index = (r+rect.y) * width + (c+rect.x);
			int rIndex = r * rect.width + c;
			mask[klabels[index]] = GC_PR_FGD;
		}
	}

}



//通过k-means算法来初始化背景GMM和前景GMM模型
/*
  Initialize GMM background and foreground models using kmeans algorithm.
*/
void SuperpixelGrabCut::initGMMs(vector<Vec6d>& superpixels, vector<vector<int> >& superarcs, vector<int>& spMask, GMM& bgdGMM, GMM& fgdGMM )
{
    const int kMeansItCount = 10;  //迭代次数
    const int kMeansType = KMEANS_PP_CENTERS; //Use kmeans++ center initialization by Arthur and Vassilvitskii

	//记录背景和前景的像素样本集中每个像素对应GMM的哪个高斯模型，论文中的kn
	Mat _bgdLabels, _fgdLabels;
	vector<Vec3f> bgdSamples, fgdSamples;//背景和前景的像素样本集

	for(int i = 0; i < spMask.size(); i++)
	{
		if(spMask[i] == GC_BGD || spMask[i] == GC_PR_BGD)
			bgdSamples.push_back((Vec3f)Vec6d_to_Vec3d(superpixels[i]));
		else 
			fgdSamples.push_back((Vec3f)Vec6d_to_Vec3d(superpixels[i]));
	}

	//kmeans中参数_bgdSamples为：每行一个样本
	//kmeans的输出为bgdLabels，里面保存的是输入样本集中每一个样本对应的类标签（样本聚为componentsCount类后）

	Mat _bgdSamples( (int)bgdSamples.size(), 3, CV_32FC1, &bgdSamples[0][0] );
    kmeans( _bgdSamples, GMM::componentsCount, _bgdLabels,
            TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType );
    Mat _fgdSamples( (int)fgdSamples.size(), 3, CV_32FC1, &fgdSamples[0][0] );
    kmeans( _fgdSamples, GMM::componentsCount, _fgdLabels,
            TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType );


    //经过上面的步骤后，每个像素所属的高斯模型就确定的了，那么就可以估计GMM中每个高斯模型的参数了。
	bgdGMM.initLearning();	//初始化某些变量=0
    for( int i = 0; i < (int)bgdSamples.size(); i++ )
        bgdGMM.addSample( _bgdLabels.at<int>(i,0), bgdSamples[i] );
    bgdGMM.endLearning();	//计算参数 和 计算某些矩阵等信息

    fgdGMM.initLearning();
    for( int i = 0; i < (int)fgdSamples.size(); i++ )
        fgdGMM.addSample( _fgdLabels.at<int>(i,0), fgdSamples[i] );
    fgdGMM.endLearning();
}

//论文中：迭代最小化算法step 1：为每个像素分配GMM中所属的高斯模型，kn保存在Mat compIdxs中（没区分是前景还是背景里的高斯分量，但可通过mask判断前景还是背景）
/*
  Assign GMMs components for each pixel.
*/
void SuperpixelGrabCut::assignGMMsComponents(vector<Vec6d>& superpixels, vector<int>& spMask, GMM& bgdGMM, GMM& fgdGMM, vector<int>& compIdxs)
{
	for(int i = 0; i < spMask.size(); i++)
	{
		if(spMask[i] == GC_BGD || spMask[i] == GC_PR_BGD)
			compIdxs[i] = bgdGMM.whichComponent(Vec6d_to_Vec3d(superpixels[i]));
		else
			compIdxs[i] = fgdGMM.whichComponent(Vec6d_to_Vec3d(superpixels[i]));
	}
}

//论文中：迭代最小化算法step 2：从每个高斯模型的像素样本集中学习每个高斯模型的参数
/*
  Learn GMMs parameters.
*/

void SuperpixelGrabCut::learnGMMs( vector<Vec6d>& superpixels, vector<int>& spMask, vector<int>& compIdxs,  GMM& bgdGMM, GMM& fgdGMM )
{
	//相关变量初始化=0
    bgdGMM.initLearning();
    fgdGMM.initLearning();

	//分配像素到前景背景样本里
	for(int ci = 0; ci < GMM::componentsCount; ci++)
	{
		for(int i = 0; i < spMask.size(); i++)
		{
			if(compIdxs[i] == ci)
			{
				if(spMask[i] == GC_BGD || spMask[i] == GC_PR_BGD)
					bgdGMM.addSample(ci, Vec6d_to_Vec3d(superpixels[i]));
				else 
					fgdGMM.addSample(ci, Vec6d_to_Vec3d(superpixels[i]));
			}
		}
	}

	//计算模型参数 和某些矩阵等信息
    bgdGMM.endLearning();
    fgdGMM.endLearning();
}

//通过计算得到的能量项构建图，图的顶点为像素点，图的边由两部分构成，
//一类边是：每个顶点与Sink汇点t（代表背景）和源点Source（代表前景）连接的边，
//这类边的权值通过Gibbs能量项的第一项能量项来表示。
//另一类边是：每个顶点与其邻域顶点连接的边，这类边的权值通过Gibbs能量项的第二项能量项来表示。
//lambda t-link权重=[0,lambda]
/*
  Construct GCGraph
*/
void SuperpixelGrabCut::constructGCGraph(vector<Vec6d>& superpixels, vector<int>& spMask, GMM& bgdGMM, GMM& fgdGMM, double lambda, vector<vector<int> >& superarcs, vector<vector<double> >& spNWeights, vector<int>& vtxs, int sideNum, GCGraph<double>& graph)
{
	int vtxCount = superpixels.size();
	int edgeCount = 2*sideNum;
	graph.create(vtxCount, edgeCount);

	vtxs.resize(vtxCount);
	for(int i = 0; i < vtxCount; i++)
	{
		// add node
		int vtxIdx = graph.addVtx();  //返回这个顶点在图中的索引
		vtxs[i] = vtxIdx;
		Vec3b color = Vec6d_to_Vec3d(superpixels[i]);

		//t-link
		double fromSource, toSink;
		if( spMask[i] == GC_PR_BGD || spMask[i] == GC_PR_FGD )
		{
			//double sizeRate = (double)superpixels[i][3] / (10*10);

			//对每一个像素计算其作为背景像素或者前景像素的第一个能量项，作为分别与t和s点的连接权值
			fromSource = -log( bgdGMM(color) );	//-log(像素属于背景模型概率)
			toSink = -log( fgdGMM(color) );		//-log(像素属于前景模型概率)
		}
		else if( spMask[i] == GC_BGD )
		{
			//对于确定为背景的像素点，它与Source点（前景）的连接为0，与Sink点的连接为lambda
			fromSource = 0;
			toSink = lambda;
		}
		else // GC_FGD
		{
			fromSource = lambda;
			toSink = 0;
		}
		//设置该顶点vtxIdx分别与Source点和Sink点的连接权值  t-link
		graph.addTermWeights( vtxIdx, fromSource, toSink );

	}


	// set n-weights  n-links
	//计算两个邻域顶点之间连接的权值。
	//也即计算Gibbs能量的第二个能量项（平滑项）
	vector<int> oneLine(spNWeights.size(), 0);
	vector<vector<int> > allWeights(spNWeights.size(), oneLine);
	for(int i = 0; i < spNWeights.size(); i++)
	{
		for(int j = 0; j < spNWeights[i].size(); j++)
		{
			int start = i;
			int end = superarcs[i][j];
			allWeights[start][end] = spNWeights[i][j];
		}
	}

	for(int i = 0; i < spNWeights.size(); i++)
	{
		for(int j = 0; j < spNWeights[i].size(); j++)
		{
			int start = i;
			int end = superarcs[i][j];
			if(start >= end) continue;
			int valueStart = allWeights[start][end];
			int valueEnd = allWeights[end][start];
			graph.addEdges(vtxs[start], vtxs[end], valueStart, valueEnd);
		}
	}


}

//论文中：迭代最小化算法step 3：分割估计：最小割或者最大流算法
/*
  Estimate segmentation using MaxFlow algorithm
*/
void SuperpixelGrabCut::estimateSegmentation( GCGraph<double>& graph, vector<int>& spMask, vector<int> vtxs)
{
    //通过最大流算法确定图的最小割，也即完成图像的分割
	graph.maxFlow();

	for(int i = 0; i < spMask.size(); i++)
	{
		//只更新PR标记
		if(spMask[i] == GC_PR_BGD || spMask[i] == GC_PR_FGD)
		{
			if( graph.inSourceSegment(vtxs[i]) )
				spMask[i] = GC_PR_FGD;
			else
				spMask[i] = GC_PR_BGD;
		}
	}

}

//最后的成果：提供给外界使用的伟大的API：grabCut 
/*
****参数说明：
	img——待分割的源图像，必须是8位3通道（CV_8UC3）图像，在处理的过程中不会被修改；
	mask——掩码图像，如果使用掩码进行初始化，那么mask保存初始化掩码信息；在执行分割
		的时候，也可以将用户交互所设定的前景与背景保存到mask中，然后再传入grabCut函
		数；在处理结束之后，mask中会保存结果。mask只能取以下四种值：
		GCD_BGD（=0），背景；
		GCD_FGD（=1），前景；
		GCD_PR_BGD（=2），可能的背景；
		GCD_PR_FGD（=3），可能的前景。
		如果没有手工标记GCD_BGD或者GCD_FGD，那么结果只会有GCD_PR_BGD或GCD_PR_FGD；
	rect——用于限定需要进行分割的图像范围，只有该矩形窗口内的图像部分才被处理；
	bgdModel——背景模型，如果为null，函数内部会自动创建一个bgdModel；bgdModel必须是
		单通道浮点型（CV_32FC1）图像，且行数只能为1，列数只能为13x5；  保存有高斯混合模型的所有参数
	fgdModel——前景模型，如果为null，函数内部会自动创建一个fgdModel；fgdModel必须是
		单通道浮点型（CV_32FC1）图像，且行数只能为1，列数只能为13x5；
	iterCount——迭代次数，必须大于0；
	mode——用于指示grabCut函数进行什么操作，可选的值有：
		GC_INIT_WITH_RECT（=0），用矩形窗初始化GrabCut；
		GC_INIT_WITH_MASK（=1），用掩码图像初始化GrabCut；
		GC_EVAL（=2），执行分割。
*/
void SuperpixelGrabCut::grabCut_ForSuperpixels(vector<int>& klabels, vector<Vec6d> superpixels, vector<vector<int> >& superarcs, double beta, vector<int>& spMask,GMM& fgdGMM, GMM& bgdGMM,  int iterCount, int mode)
{
	int spSize = superpixels.size();


	//保存像素所属于的高斯分量（没区分是前景还是背景里的高斯分量）
	vector<int> compIdxs(spSize,0);

	//若用户没有给出混合模型的参数，则需要学习模型参数（initGMMs）
	if(mode == GC_INIT_WITH_RECT)	
	{
		return;	//error
	}
    if(mode == GC_INIT_WITH_MASK )
    {//初次分割
		//用kmeans++确定每个像素属于前景还是背景混合模型里具体的哪个高斯分量，然后用这些样本点估算前景背景混合模型的所有参数
        initGMMs(superpixels, superarcs, spMask, bgdGMM, fgdGMM );
    }


    if( iterCount <= 0)
        return;

	//若用户给出了混合模型参数，则只检查下，不调用(initGMMs)
    if( mode == GC_EVAL )
        checkMask(superpixels, spMask );

    const double gamma = 50;		//论文中gamma (希腊字母第三个)
    const double lambda = 9*gamma;	//论文中lambda (希腊字母第十一个)
    //const double beta = calcBeta( img );	//论文中beta（希腊字母第二个），扩大缩小图形对比度（把太大的缩小，把太小的扩大）

	//计算n-link
	vector<vector<double> > spNWeights;
	vector<int> vtxs;			//记录超像素标记与graph中点标记的对应关系
	int sideNum = 0;
    //Mat leftW, upleftW, upW, uprightW;
	calcNWeights(superpixels, superarcs, spNWeights, sideNum, beta, gamma);

	//前面先用kmeans++算法确定像素聚类情况，进而确定分量，再才能初始化混合模型参数，然后计算图中边的情况（边情况不会发生变化）
	//之后迭代的（1.根据混合模型计算所有像素所属高斯分量，2.重新计算混合模型参数，3.计算图割，4.结果保存在mask）
    for( int i = 0; i < iterCount; i++ )
    {
        GCGraph<double> graph;
		//计算每个像素所属哪个高斯分量
		assignGMMsComponents( superpixels, spMask, bgdGMM, fgdGMM, compIdxs);
		//计算混合模型的所有参数/更新参数（和之前计算出的参数有差异）
        learnGMMs( superpixels, spMask, compIdxs, bgdGMM, fgdGMM );
		//构造图, 添加 t-link n-link
		constructGCGraph(superpixels, spMask, bgdGMM, fgdGMM, lambda, superarcs, spNWeights, vtxs, sideNum, graph);
		//使用最大流算法完成分割，然后更新mask(永远不会改变用户指定的部分)
        estimateSegmentation(graph, spMask, vtxs);
    }
}




/////////////////////////////



void PixelsGrabCut::calcNWeights( const Mat& img, Mat& leftW, Mat& upleftW, Mat& upW, 
							Mat& uprightW, double beta, double gamma )
{
    //gammaDivSqrt2相当于公式（4）中的gamma * dis(i,j)^(-1)，那么可以知道，
	//当i和j是垂直或者水平关系时，dis(i,j)=1，当是对角关系时，dis(i,j)=sqrt(2.0f)。
	//具体计算时，看下面就明白了
	const double gammaDivSqrt2 = gamma / std::sqrt(2.0f);
	//每个方向的边的权值通过一个和图大小相等的Mat来保存
    leftW.create( img.rows, img.cols, CV_64FC1 );
    upleftW.create( img.rows, img.cols, CV_64FC1 );
    upW.create( img.rows, img.cols, CV_64FC1 );
    uprightW.create( img.rows, img.cols, CV_64FC1 );
    for( int y = 0; y < img.rows; y++ )
    {
        for( int x = 0; x < img.cols; x++ )
        {
            Vec3d color = img.at<Vec3b>(y,x);
            if( x-1>=0 ) // left  //避免图的边界
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y,x-1);
                leftW.at<double>(y,x) = gamma * exp(-beta*diff.dot(diff));
            }
            else
                leftW.at<double>(y,x) = 0;
            if( x-1>=0 && y-1>=0 ) // upleft
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x-1);
                upleftW.at<double>(y,x) = gammaDivSqrt2 * exp(-beta*diff.dot(diff));
            }
            else
                upleftW.at<double>(y,x) = 0;
            if( y-1>=0 ) // up
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x);
                upW.at<double>(y,x) = gamma * exp(-beta*diff.dot(diff));
            }
            else
                upW.at<double>(y,x) = 0;
            if( x+1<img.cols && y-1>=0 ) // upright
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x+1);
                uprightW.at<double>(y,x) = gammaDivSqrt2 * exp(-beta*diff.dot(diff));
            }
            else
                uprightW.at<double>(y,x) = 0;
        }
    }
}

//检查mask的正确性。mask为通过用户交互或者程序设定的，它是和图像大小一样的单通道灰度图，
//每个像素只能取GC_BGD or GC_FGD or GC_PR_BGD or GC_PR_FGD 四种枚举值，分别表示该像素
//（用户或者程序指定）属于背景、前景、可能为背景或者可能为前景像素。具体的参考：
//ICCV2001“Interactive Graph Cuts for Optimal Boundary & Region Segmentation of Objects in N-D Images”
//Yuri Y. Boykov Marie-Pierre Jolly 
/*
  Check size, type and element values of mask matrix.
 */
void PixelsGrabCut::checkMask( const Mat& img, const Mat& mask )
{
    if( mask.empty() )
        CV_Error( CV_StsBadArg, "mask is empty" );
    if( mask.type() != CV_8UC1 )
        CV_Error( CV_StsBadArg, "mask must have CV_8UC1 type" );
    if( mask.cols != img.cols || mask.rows != img.rows )
        CV_Error( CV_StsBadArg, "mask must have as many rows and cols as img" );
    for( int y = 0; y < mask.rows; y++ )
    {
        for( int x = 0; x < mask.cols; x++ )
        {
            uchar val = mask.at<uchar>(y,x);
            if( val!=GC_BGD && val!=GC_FGD && val!=GC_PR_BGD && val!=GC_PR_FGD )
                CV_Error( CV_StsBadArg, "mask element value must be equel"
                    "GC_BGD or GC_FGD or GC_PR_BGD or GC_PR_FGD" );
        }
    }
}

//通过用户框选目标rect来创建mask，rect外的全部作为背景，设置为GC_BGD，
//rect内的设置为 GC_PR_FGD（可能为前景）
/*
  Initialize mask using rectangular.
*/
void PixelsGrabCut::initMaskWithRect( Mat& mask, Size imgSize, Rect rect )
{
    mask.create( imgSize, CV_8UC1 );
    mask.setTo( GC_BGD );

    rect.x = max(0, rect.x);
    rect.y = max(0, rect.y);
    rect.width = min(rect.width, imgSize.width-rect.x);
    rect.height = min(rect.height, imgSize.height-rect.y);

    (mask(rect)).setTo( Scalar(GC_PR_FGD) );
}

//论文中：迭代最小化算法step 1：为每个像素分配GMM中所属的高斯模型，kn保存在Mat compIdxs中（没区分是前景还是背景里的高斯分量，但可通过mask判断前景还是背景）
/*
  Assign GMMs components for each pixel.
*/
void PixelsGrabCut::assignGMMsComponents( const Mat& img, const Mat& mask, const GMM& bgdGMM, 
									const GMM& fgdGMM, Mat& compIdxs )
{
    Point p;
    for( p.y = 0; p.y < img.rows; p.y++ )
    {
        for( p.x = 0; p.x < img.cols; p.x++ )
        {
            Vec3d color = img.at<Vec3b>(p);
			//通过mask来判断该像素属于背景像素还是前景像素，再判断它属于前景或者背景GMM中的哪个高斯分量
			//在计算混合模型参数时，使用kmeans++聚类计算过每个像素所属的高斯分量，而这里通过计算像素在不同高斯分量里的概率来判断。

            compIdxs.at<int>(p) = mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD ? bgdGMM.whichComponent(color) : fgdGMM.whichComponent(color);


        }
    }
}

//论文中：迭代最小化算法step 2：从每个高斯模型的像素样本集中学习每个高斯模型的参数
/*
  Learn GMMs parameters.
*/
void PixelsGrabCut::learnGMMs( const Mat& img, const Mat& mask, const Mat& compIdxs, GMM& bgdGMM, GMM& fgdGMM )
{
	//相关变量初始化=0
    bgdGMM.initLearning();
    fgdGMM.initLearning();

	//分配像素到前景背景样本里
    Point p;
    for( int ci = 0; ci < GMM::componentsCount; ci++ )
    {
        for( p.y = 0; p.y < img.rows; p.y++ )
        {
            for( p.x = 0; p.x < img.cols; p.x++ )
            {
                if( compIdxs.at<int>(p) == ci )
                {
                    if( mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD )
                        bgdGMM.addSample( ci, img.at<Vec3b>(p) );
                    else
                        fgdGMM.addSample( ci, img.at<Vec3b>(p) );
                }
            }
        }
    }

	//计算模型参数 和某些矩阵等信息
    bgdGMM.endLearning();
    fgdGMM.endLearning();
}

//通过计算得到的能量项构建图，图的顶点为像素点，图的边由两部分构成，
//一类边是：每个顶点与Sink汇点t（代表背景）和源点Source（代表前景）连接的边，
//这类边的权值通过Gibbs能量项的第一项能量项来表示。
//另一类边是：每个顶点与其邻域顶点连接的边，这类边的权值通过Gibbs能量项的第二项能量项来表示。
//lambda t-link权重=[0,lambda]
/*
  Construct GCGraph
*/
void PixelsGrabCut::constructGCGraph( const Mat& img, const Mat& mask, const GMM& bgdGMM, const GMM& fgdGMM, double lambda,
                       const Mat& leftW, const Mat& upleftW, const Mat& upW, const Mat& uprightW,
                       GCGraph<double>& graph )
{
    int vtxCount = img.cols*img.rows;  //顶点数，每一个像素是一个顶点

	//(8领域，上下左右+左上、坐下、右上、右下等普通边(n-link)，未包含到source,sink的特殊边(t-link)) 当做有向图处理，边数量是无向图的2倍
    int edgeCount = 2*(4*vtxCount/*left,upLeft,up,upRight*/ - 3*(img.cols + img.rows)/*多计算的边界缺失边*/ + 2/*重复的边*/);  //边数，需要考虑图边界的边的缺失
    //通过顶点数和边数创建图。这些类型声明和函数定义请参考gcgraph.hpp
	graph.create(vtxCount, edgeCount);
    Point p;
    for( p.y = 0; p.y < img.rows; p.y++ )
    {
        for( p.x = 0; p.x < img.cols; p.x++)
        {
            // add node
            int vtxIdx = graph.addVtx();  //返回这个顶点在图中的索引
            Vec3b color = img.at<Vec3b>(p);

            // set t-weights			
            //计算每个顶点与Sink汇点t（代表背景）和源点Source（代表前景）连接的权值。
			//也即计算Gibbs能量（每一个像素点作为背景像素或者前景像素）的第一个能量项
			double fromSource, toSink;
            if( mask.at<uchar>(p) == GC_PR_BGD || mask.at<uchar>(p) == GC_PR_FGD )
            {
                //对每一个像素计算其作为背景像素或者前景像素的第一个能量项，作为分别与t和s点的连接权值
				fromSource = -log( bgdGMM(color) );	//-log(像素属于背景模型概率)
                toSink = -log( fgdGMM(color) );		//-log(像素属于前景模型概率)
            }
            else if( mask.at<uchar>(p) == GC_BGD )
            {
                //对于确定为背景的像素点，它与Source点（前景）的连接为0，与Sink点的连接为lambda
				fromSource = 0;
                toSink = lambda;
            }
            else // GC_FGD
            {
                fromSource = lambda;
                toSink = 0;
            }
			//设置该顶点vtxIdx分别与Source点和Sink点的连接权值
            graph.addTermWeights( vtxIdx, fromSource, toSink );

            // set n-weights  n-links
            //计算两个邻域顶点之间连接的权值。
			//也即计算Gibbs能量的第二个能量项（平滑项）
			if( p.x>0 )
            {
                double w = leftW.at<double>(p);
                graph.addEdges( vtxIdx, vtxIdx-1, w, w );
            }
            if( p.x>0 && p.y>0 )
            {
                double w = upleftW.at<double>(p);
                graph.addEdges( vtxIdx, vtxIdx-img.cols-1, w, w );
            }
            if( p.y>0 )
            {
                double w = upW.at<double>(p);
                graph.addEdges( vtxIdx, vtxIdx-img.cols, w, w );
            }
            if( p.x<img.cols-1 && p.y>0 )
            {
                double w = uprightW.at<double>(p);
                graph.addEdges( vtxIdx, vtxIdx-img.cols+1, w, w );
            }
        }
    }
}

//论文中：迭代最小化算法step 3：分割估计：最小割或者最大流算法
/*
  Estimate segmentation using MaxFlow algorithm
*/
void PixelsGrabCut::estimateSegmentation( GCGraph<double>& graph, Mat& mask )
{
    //通过最大流算法确定图的最小割，也即完成图像的分割
	graph.maxFlow();
    Point p;
    for( p.y = 0; p.y < mask.rows; p.y++ )
    {
        for( p.x = 0; p.x < mask.cols; p.x++ )
        {
            //通过图分割的结果来更新mask，即最后的图像分割结果。注意的是，永远都
			//不会更新用户指定为背景或者前景的像素
			if( mask.at<uchar>(p) == GC_PR_BGD || mask.at<uchar>(p) == GC_PR_FGD )
            {
                if( graph.inSourceSegment( p.y*mask.cols+p.x /*vertex index*/ ) )
                    mask.at<uchar>(p) = GC_PR_FGD;
                else
                    mask.at<uchar>(p) = GC_PR_BGD;
            }
        }
    }
}


double PixelsGrabCut::calcBeta( const Mat& img )
{
    double beta = 0;
    for( int y = 0; y < img.rows; y++ )
    {
        for( int x = 0; x < img.cols; x++ )
        {
			//累积所有相邻像素差的点乘
			//计算四个方向邻域两像素的差别，也就是欧式距离或者说二阶范数   只计算图像八领域的一半（left upleft up upright），防止重复计算
			//（当所有像素都算完后，就相当于计算八邻域的像素差了）
            Vec3d color = img.at<Vec3b>(y,x);
            if( x>0 ) // color-left  >0的判断是为了避免在图像边界的时候还计算，导致越界
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y,x-1);
                beta += diff.dot(diff);  //矩阵的点乘，也就是各个元素平方的和
            }
            if( y>0 && x>0 ) // color-upleft
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x-1);
                beta += diff.dot(diff);
            }
            if( y>0 ) // color-up
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x);
                beta += diff.dot(diff);
            }
            if( y>0 && x<img.cols-1) // color-upright
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x+1);
                beta += diff.dot(diff);
            }
        }
    }
    if( beta <= std::numeric_limits<double>::epsilon() )
        beta = 0;
    else
        beta = 1.f / (2 * beta/(4*img.cols*img.rows - 3*img.cols - 3*img.rows + 2) ); //论文公式（5）   beta= 1.0f / (2 * beta/(之前累积点乘的相邻像素的边数));   无向图8领域

    return beta;
}


//最后的成果：提供给外界使用的伟大的API：grabCut 
/*
****参数说明：
	img——待分割的源图像，必须是8位3通道（CV_8UC3）图像，在处理的过程中不会被修改；
	mask——掩码图像，如果使用掩码进行初始化，那么mask保存初始化掩码信息；在执行分割
		的时候，也可以将用户交互所设定的前景与背景保存到mask中，然后再传入grabCut函
		数；在处理结束之后，mask中会保存结果。mask只能取以下四种值：
		GCD_BGD（=0），背景；
		GCD_FGD（=1），前景；
		GCD_PR_BGD（=2），可能的背景；
		GCD_PR_FGD（=3），可能的前景。
		如果没有手工标记GCD_BGD或者GCD_FGD，那么结果只会有GCD_PR_BGD或GCD_PR_FGD；
	rect——用于限定需要进行分割的图像范围，只有该矩形窗口内的图像部分才被处理；
	bgdModel——背景模型，如果为null，函数内部会自动创建一个bgdModel；bgdModel必须是
		单通道浮点型（CV_32FC1）图像，且行数只能为1，列数只能为13x5；  保存有高斯混合模型的所有参数
	fgdModel——前景模型，如果为null，函数内部会自动创建一个fgdModel；fgdModel必须是
		单通道浮点型（CV_32FC1）图像，且行数只能为1，列数只能为13x5；
	iterCount——迭代次数，必须大于0；
	mode——用于指示grabCut函数进行什么操作，可选的值有：
		GC_INIT_WITH_RECT（=0），用矩形窗初始化GrabCut；
		GC_INIT_WITH_MASK（=1），用掩码图像初始化GrabCut；
		GC_EVAL（=2），执行分割。
*/
void PixelsGrabCut::grabCut_ForPixel( Mat& img, Mat& mask, Rect& rect, GMM& fgdGMM, GMM& bgdGMM, int iterCount, bool isFirst)
{

    if( img.empty() )
        CV_Error( CV_StsBadArg, "image is empty" );
    if( img.type() != CV_8UC3 )
        CV_Error( CV_StsBadArg, "image mush have CV_8UC3 type" );


	//保存像素所属于的高斯分量（没区分是前景还是背景里的高斯分量）
    Mat compIdxs( img.size(), CV_32SC1 );

	//if(isFirst)
	//{
	//	//initGMMs( img, mask, bgdGMM, fgdGMM );
	//}

	////若用户没有给出混合模型的参数，则需要学习模型参数（initGMMs）
 //   if( mode == GC_INIT_WITH_RECT || mode == GC_INIT_WITH_MASK )
 //   {
 //       if( mode == GC_INIT_WITH_RECT )
 //           initMaskWithRect( mask, img.size(), rect );	//rect外=GC_BGD  rect内=GC_PR_FGD
 //       else // flag == GC_INIT_WITH_MASK
 //           checkMask( img, mask );	//检查mask里值的范围：GC_BGD，GC_FGD, GC_PR_BGD, GC_PR_FGD
	//	//用kmeans++确定每个像素属于前景还是背景混合模型里具体的哪个高斯分量，然后用这些样本点估算前景背景混合模型的所有参数
 //       initGMMs( img, mask, bgdGMM, fgdGMM );
 //   }
	//initMaskWithRect( mask, img.size(), rect );



    if( iterCount <= 0)
        return;

	//若用户给出了混合模型参数，则只检查下，不调用(initGMMs)
    checkMask( img, mask );

    const double gamma = 50;		//论文中gamma (希腊字母第三个)
    const double lambda = 9*gamma;	//论文中lambda (希腊字母第十一个)
    const double beta = calcBeta( img );	//论文中beta（希腊字母第二个），扩大缩小图形对比度（把太大的缩小，把太小的扩大）

	//计算n-link
    Mat leftW, upleftW, upW, uprightW;
    calcNWeights( img, leftW, upleftW, upW, uprightW, beta, gamma );

	//前面先用kmeans++算法确定像素聚类情况，进而确定分量，再才能初始化混合模型参数，然后计算图中边的情况（边情况不会发生变化）
	//之后迭代的（1.根据混合模型计算所有像素所属高斯分量，2.重新计算混合模型参数，3.计算图割，4.结果保存在mask）
    for( int i = 0; i < iterCount; i++ )
    {
        GCGraph<double> graph;

		//Mat mask2 = mask.clone();
		//Point p;
		//int a0 = 0;
		//int a1 = 255/3;
		//int a2 = 255/3*2;
		//int a3 = 255;
		//for(p.y = 0; p.y < mask2.rows; p.y++)
		//{
		//	for(p.x = 0; p.x < mask2.cols; p.x++)
		//	{
		//		if(mask2.at<uchar>(p) == 1)
		//			mask2.at<uchar>(p) = a1;
		//		if(mask2.at<uchar>(p) == 2)
		//			mask2.at<uchar>(p) = a2;
		//		if(mask2.at<uchar>(p) == 3)
		//			mask2.at<uchar>(p) = a3;
		//	}
		//}

		//imshow("fullAA", mask2);



		//double begin = (double)getTickCount();
		////...
		//double end = (double)getTickCount();
		//double time = (end-begin)/getTickFrequency()



		//计算每个像素所属哪个高斯分量
        assignGMMsComponents( img, mask, bgdGMM, fgdGMM, compIdxs );
		//计算混合模型的所有参数/更新参数（和之前计算出的参数有差异）
        learnGMMs( img, mask, compIdxs, bgdGMM, fgdGMM );
		//构造图, 添加 t-link n-link
        constructGCGraph(img, mask, bgdGMM, fgdGMM, lambda, leftW, upleftW, upW, uprightW, graph );
		//使用最大流算法完成分割，然后更新mask(永远不会改变用户指定的部分)
        estimateSegmentation( graph, mask );
    }
}
