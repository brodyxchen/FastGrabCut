

#include "Superpixel.h"

const int dx4[4] = {-1,  0,  1,  0};
const int dy4[4] = { 0, -1,  0,  1};
const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};



SLICO::SLICO()
{
	m_lvec = NULL;
	m_avec = NULL;
	m_bvec = NULL;

	m_lvecvec = NULL;
	m_avecvec = NULL;
	m_bvecvec = NULL;
}

SLICO::~SLICO()
{
	if(m_lvec) delete [] m_lvec;
	if(m_avec) delete [] m_avec;
	if(m_bvec) delete [] m_bvec;


	if(m_lvecvec)
	{
		for( int d = 0; d < m_depth; d++ ) delete [] m_lvecvec[d];
		delete [] m_lvecvec;
	}
	if(m_avecvec)
	{
		for( int d = 0; d < m_depth; d++ ) delete [] m_avecvec[d];
		delete [] m_avecvec;
	}
	if(m_bvecvec)
	{
		for( int d = 0; d < m_depth; d++ ) delete [] m_bvecvec[d];
		delete [] m_bvecvec;
	}
}


void SLICO::RGB2XYZ(
	const int&		sR,
	const int&		sG,
	const int&		sB,
	double&			X,
	double&			Y,
	double&			Z)
{
	double R = sR/255.0;
	double G = sG/255.0;
	double B = sB/255.0;

	double r, g, b;

	if(R <= 0.04045)	r = R/12.92;
	else				r = pow((R+0.055)/1.055,2.4);
	if(G <= 0.04045)	g = G/12.92;
	else				g = pow((G+0.055)/1.055,2.4);
	if(B <= 0.04045)	b = B/12.92;
	else				b = pow((B+0.055)/1.055,2.4);

	X = r*0.4124564 + g*0.3575761 + b*0.1804375;
	Y = r*0.2126729 + g*0.7151522 + b*0.0721750;
	Z = r*0.0193339 + g*0.1191920 + b*0.9503041;
}


void SLICO::RGB2LAB(const int& sR, const int& sG, const int& sB, double& lval, double& aval, double& bval)
{
	//------------------------
	// sRGB to XYZ conversion
	//------------------------
	double X, Y, Z;
	RGB2XYZ(sR, sG, sB, X, Y, Z);

	//------------------------
	// XYZ to LAB conversion
	//------------------------
	double epsilon = 0.008856;	//actual CIE standard
	double kappa   = 903.3;		//actual CIE standard

	double Xr = 0.950456;	//reference white
	double Yr = 1.0;		//reference white
	double Zr = 1.088754;	//reference white

	double xr = X/Xr;
	double yr = Y/Yr;
	double zr = Z/Zr;

	double fx, fy, fz;
	if(xr > epsilon)	fx = pow(xr, 1.0/3.0);
	else				fx = (kappa*xr + 16.0)/116.0;
	if(yr > epsilon)	fy = pow(yr, 1.0/3.0);
	else				fy = (kappa*yr + 16.0)/116.0;
	if(zr > epsilon)	fz = pow(zr, 1.0/3.0);
	else				fz = (kappa*zr + 16.0)/116.0;

	lval = 116.0*fy-16.0;
	aval = 500.0*(fx-fy);
	bval = 200.0*(fy-fz);
}


void SLICO::DoRGBtoLABConversion(
	const unsigned int*&		ubuff,
	double*&					lvec,
	double*&					avec,
	double*&					bvec)
{
	int sz = m_width*m_height;
	lvec = new double[sz];
	avec = new double[sz];
	bvec = new double[sz];

	for( int j = 0; j < sz; j++ )
	{
		int r = (ubuff[j] >> 16) & 0xFF;
		int g = (ubuff[j] >>  8) & 0xFF;
		int b = (ubuff[j]      ) & 0xFF;

		RGB2LAB( r, g, b, lvec[j], avec[j], bvec[j] );
	}
}



//绘制边界
void SLICO::DrawContoursAroundSegments(Mat& mat, vector<int> klabels, Scalar color)
{
	const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};

	Vec3d cvColor;
	cvColor[0] = 255;//color[0];
	cvColor[1] = 255;//color[1];
	cvColor[2] = 255;//color[2];

	int channels = mat.channels();
	int nRows = mat.rows;
	int nCols = mat.cols * channels;
	int height = mat.rows;
	int width = mat.cols * channels;


	
	int sz = mat.rows * mat.cols;
	vector<bool> istaken(sz, false);


	if(mat.isContinuous())
	{
		nCols *= nRows;
		nRows = 1;

		uchar* data = mat.ptr<uchar>(0);
		for(int i = 0; i < nCols; i+=channels)
		{
			int r = i / width;
			int c = i % width;

			int np(0);
			for(int k = 0; k < 8; k++)
			{
				int newr = r + dy8[k];
				int newc = c + dx8[k]*channels;
				if( (newc >= 0 && newc < width) && (newr >= 0 && newr < height) )
				{
					int newIndex = newr * width + newc;
					if(false == istaken[newIndex/channels])
					{
						if(klabels[i/channels] != klabels[newIndex/channels])
						{
							np++;
						}
					}

				}
			}
			if(np > 1)
			{
				*(data+i) = cvColor[0];
				*(data+i+1) = cvColor[1];
				*(data+i+2) = cvColor[2];
				istaken[i/channels] = true;
			}

		}

	}else
	{
		uchar* data;
		for(int r = 0; r < nRows; r++)
		{
			data = mat.ptr<uchar>(r);
			for(int c = 0; c < nCols; c+=channels)
			{
				int index = r * width + c;

				int np(0);
				for(int k = 0; k < 8; k++)
				{
					int newr = r + dy8[k];
					int newc = c + dx8[k]*channels;
					if( (newc >= 0 && newc < width) && (newr >= 0 && newr < height) )
					{
						int newIndex = newr * width + newc;
						if(false == istaken[newIndex/channels])
						{
							if(klabels[index/channels] != klabels[newIndex/channels])
							{
								np++;
							}
						}

					}
				}
				if(np > 1)
				{
					*(data+c) = cvColor[0];
					*(data+c+1) = cvColor[1];
					*(data+c+2) = cvColor[2];
					istaken[index/channels] = true;
				}
			}
		}
	}


	/*
	if(mat.isContinuous())
	{
		nCols *= nRows;
		nRows = 1;

		uchar* data = mat.ptr<uchar>(0);
		for(int i = 0; i < nCols; i+=channels)
		{
			int r = i / width;
			int c = i % width;
			bool isSidePixel = false;
			for(int k = 0; k < 8; k++)
			{
				int newr = r + dy8[k];
				int newc = c + dx8[k]*channels;
				if( (newc >= 0 && newc < width) && (newr >= 0 && newr < height) )
				{
					int newIndex = newr * width + newc;
					if(klabels[i/channels] != klabels[newIndex/channels])
					{
						isSidePixel = true;
						break;
					}
				}
			}
			if(isSidePixel)
			{
				*(data+i) = cvColor[0];
				*(data+i+1) = cvColor[1];
				*(data+i+2) = cvColor[2];
			}

		}


	}else
	{
		uchar* data;
		for(int r = 0; r < nRows; r++)
		{
			data = mat.ptr<uchar>(r);
			for(int c = 0; c < nCols; c+=channels)
			{
				int index = r * width + c;
				bool isSidePixel = false;
				for(int k = 0; k < 8; k++)
				{
					int newr = r + dy8[k];
					int newc = c + dx8[k]*channels;
					if( (newc >= 0 && newc < width) && (newr >= 0 && newr < height) )
					{
						int newIndex = newr * width + newc;
						if(klabels[index/channels] != klabels[newIndex/channels])
						{
							isSidePixel = true;
							break;
						}
					}
				}
				if(isSidePixel)
				{
					*(data+c) = cvColor[0];
					*(data+c+1) = cvColor[1];
					*(data+c+2) = cvColor[2];
				}
			}
		}
	}
	*/




}


void SLICO::DetectLabEdges(
	const double*				lvec,
	const double*				avec,
	const double*				bvec,
	const int&					width,
	const int&					height,
	vector<double>&				edges)
{
	int sz = width*height;

	edges.resize(sz,0);
	for( int j = 1; j < height-1; j++ )
	{
		for( int k = 1; k < width-1; k++ )
		{
			int i = j*width+k;

			double dx = (lvec[i-1]-lvec[i+1])*(lvec[i-1]-lvec[i+1]) +
						(avec[i-1]-avec[i+1])*(avec[i-1]-avec[i+1]) +
						(bvec[i-1]-bvec[i+1])*(bvec[i-1]-bvec[i+1]);

			double dy = (lvec[i-width]-lvec[i+width])*(lvec[i-width]-lvec[i+width]) +
						(avec[i-width]-avec[i+width])*(avec[i-width]-avec[i+width]) +
						(bvec[i-width]-bvec[i+width])*(bvec[i-width]-bvec[i+width]);

			//edges[i] = (sqrt(dx) + sqrt(dy));
			edges[i] = (dx + dy);
		}
	}
}


void SLICO::PerturbSeeds(
	vector<double>&				kseedsl,
	vector<double>&				kseedsa,
	vector<double>&				kseedsb,
	vector<double>&				kseedsx,
	vector<double>&				kseedsy,
	const vector<double>&		edges)
{
	const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};
	
	int numseeds = kseedsl.size();

	for( int n = 0; n < numseeds; n++ )
	{
		int ox = kseedsx[n];//original x
		int oy = kseedsy[n];//original y
		int oind = oy*m_width + ox;

		int storeind = oind;
		for( int i = 0; i < 8; i++ )
		{
			int nx = ox+dx8[i];//new x
			int ny = oy+dy8[i];//new y

			if( nx >= 0 && nx < m_width && ny >= 0 && ny < m_height)
			{
				int nind = ny*m_width + nx;
				if( edges[nind] < edges[storeind])
				{
					storeind = nind;
				}
			}
		}
		if(storeind != oind)
		{
			kseedsx[n] = storeind%m_width;
			kseedsy[n] = storeind/m_width;
			kseedsl[n] = m_lvec[storeind];
			kseedsa[n] = m_avec[storeind];
			kseedsb[n] = m_bvec[storeind];
		}
	}
}


void SLICO::GetLABXYSeeds_ForGivenStepSize(
	vector<double>&				kseedsl,
	vector<double>&				kseedsa,
	vector<double>&				kseedsb,
	vector<double>&				kseedsx,
	vector<double>&				kseedsy,
	const int&					STEP,
	const bool&					perturbseeds,
	const vector<double>&		edgemag)
{
	int numseeds(0);
	int n(0);

	//int xstrips = m_width/STEP;
	//int ystrips = m_height/STEP;
	int xstrips = (0.5+double(m_width)/double(STEP));
	int ystrips = (0.5+double(m_height)/double(STEP));

	int xerr = m_width  - STEP*xstrips;
	int yerr = m_height - STEP*ystrips;

	double xerrperstrip = double(xerr)/double(xstrips);
	double yerrperstrip = double(yerr)/double(ystrips);

	int xoff = STEP/2;
	int yoff = STEP/2;
	//-------------------------
	numseeds = xstrips*ystrips;
	//-------------------------
	kseedsl.resize(numseeds);
	kseedsa.resize(numseeds);
	kseedsb.resize(numseeds);
	kseedsx.resize(numseeds);
	kseedsy.resize(numseeds);

	for( int y = 0; y < ystrips; y++ )
	{
		int ye = y*yerrperstrip;
		for( int x = 0; x < xstrips; x++ )
		{
			int xe = x*xerrperstrip;
			int i = (y*STEP+yoff+ye)*m_width + (x*STEP+xoff+xe);
			
			kseedsl[n] = m_lvec[i];
			kseedsa[n] = m_avec[i];
			kseedsb[n] = m_bvec[i];
			kseedsx[n] = (x*STEP+xoff+xe);
			kseedsy[n] = (y*STEP+yoff+ye);
			n++;
		}
	}

	
	if(perturbseeds)
	{
		PerturbSeeds(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, edgemag);
	}
}


void SLICO::GetLABXYSeeds_ForGivenK(
	vector<double>&				kseedsl,
	vector<double>&				kseedsa,
	vector<double>&				kseedsb,
	vector<double>&				kseedsx,
	vector<double>&				kseedsy,
	const int&					K,
	const bool&					perturbseeds,
	const vector<double>&		edgemag)
{
	int sz = m_width*m_height;
	double step = sqrt(double(sz)/double(K));
	int T = step;
	int xoff = step/2;
	int yoff = step/2;
	
	int n(0);int r(0);
	for( int y = 0; y < m_height; y++ )
	{
		int Y = y*step + yoff;
		if( Y > m_height-1 ) break;

		for( int x = 0; x < m_width; x++ )
		{
			//int X = x*step + xoff;//square grid
			int X = x*step + (xoff<<(r&0x1));//hex grid
			if(X > m_width-1) break;

			int i = Y*m_width + X;

			//_ASSERT(n < K);
			
			//kseedsl[n] = m_lvec[i];
			//kseedsa[n] = m_avec[i];
			//kseedsb[n] = m_bvec[i];
			//kseedsx[n] = X;
			//kseedsy[n] = Y;
			kseedsl.push_back(m_lvec[i]);
			kseedsa.push_back(m_avec[i]);
			kseedsb.push_back(m_bvec[i]);
			kseedsx.push_back(X);
			kseedsy.push_back(Y);
			n++;
		}
		r++;
	}

	if(perturbseeds)
	{
		PerturbSeeds(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, edgemag);
	}
}



void SLICO::PerformSuperpixelSegmentation_VariableSandM(
	vector<double>&				kseedsl,
	vector<double>&				kseedsa,
	vector<double>&				kseedsb,
	vector<double>&				kseedsx,
	vector<double>&				kseedsy,
	int*						klabels,
	const int&					STEP,
	const int&					NUMITR)
{
	int sz = m_width*m_height;
	const int numk = kseedsl.size();
	//double cumerr(99999.9);
	int numitr(0);

	//----------------
	int offset = STEP;
	m_size = STEP;
	if(STEP < 10) offset = STEP*1.5;
	//----------------

	vector<double> sigmal(numk, 0);
	vector<double> sigmaa(numk, 0);
	vector<double> sigmab(numk, 0);
	vector<double> sigmax(numk, 0);
	vector<double> sigmay(numk, 0);
	vector<int> clustersize(numk, 0);
	vector<double> inv(numk, 0);//to store 1/clustersize[k] values
	vector<double> distxy(sz, DBL_MAX);
	vector<double> distlab(sz, DBL_MAX);
	vector<double> distvec(sz, DBL_MAX);
	vector<double> maxlab(numk, 10*10);//THIS IS THE VARIABLE VALUE OF M, just start with 10
	vector<double> maxxy(numk, STEP*STEP);//THIS IS THE VARIABLE VALUE OF M, just start with 10

	double invxywt = 1.0/(STEP*STEP);//NOTE: this is different from how usual SLIC/LKM works

	while( numitr < NUMITR )
	{
		//------
		//cumerr = 0;
		numitr++;
		//------

		distvec.assign(sz, DBL_MAX);
		for( int n = 0; n < numk; n++ )
		{
			int y1 = max((double)0,			kseedsy[n]-offset);
			int y2 = min((double)m_height,	kseedsy[n]+offset);
			int x1 = max((double)0,			kseedsx[n]-offset);
			int x2 = min((double)m_width,	kseedsx[n]+offset);

			for( int y = y1; y < y2; y++ )
			{
				for( int x = x1; x < x2; x++ )
				{
					int i = y*m_width + x;
					_ASSERT( y < m_height && x < m_width && y >= 0 && x >= 0 );

					double l = m_lvec[i];
					double a = m_avec[i];
					double b = m_bvec[i];

					distlab[i] =	(l - kseedsl[n])*(l - kseedsl[n]) +
									(a - kseedsa[n])*(a - kseedsa[n]) +
									(b - kseedsb[n])*(b - kseedsb[n]);

					distxy[i] =		(x - kseedsx[n])*(x - kseedsx[n]) +
									(y - kseedsy[n])*(y - kseedsy[n]);

					//------------------------------------------------------------------------
					double dist = distlab[i]/maxlab[n] + distxy[i]*invxywt;//only varying m, prettier superpixels
					//double dist = distlab[i]/maxlab[n] + distxy[i]/maxxy[n];//varying both m and S
					//------------------------------------------------------------------------
					
					if( dist < distvec[i] )
					{
						distvec[i] = dist;
						klabels[i]  = n;
					}
				}
			}
		}
		//-----------------------------------------------------------------
		// Assign the max color distance for a cluster
		//-----------------------------------------------------------------
		if(0 == numitr)
		{
			maxlab.assign(numk,1);
			maxxy.assign(numk,1);
		}
		{for( int i = 0; i < sz; i++ )
		{
			if(maxlab[klabels[i]] < distlab[i]) maxlab[klabels[i]] = distlab[i];
			if(maxxy[klabels[i]] < distxy[i]) maxxy[klabels[i]] = distxy[i];
		}}
		//-----------------------------------------------------------------
		// Recalculate the centroid and store in the seed values
		//-----------------------------------------------------------------
		sigmal.assign(numk, 0);
		sigmaa.assign(numk, 0);
		sigmab.assign(numk, 0);
		sigmax.assign(numk, 0);
		sigmay.assign(numk, 0);
		clustersize.assign(numk, 0);

		for( int j = 0; j < sz; j++ )
		{
			int temp = klabels[j];
			_ASSERT(klabels[j] >= 0);
			sigmal[klabels[j]] += m_lvec[j];
			sigmaa[klabels[j]] += m_avec[j];
			sigmab[klabels[j]] += m_bvec[j];
			sigmax[klabels[j]] += (j%m_width);
			sigmay[klabels[j]] += (j/m_width);

			clustersize[klabels[j]]++;
		}

		{for( int k = 0; k < numk; k++ )
		{
			//_ASSERT(clustersize[k] > 0);
			if( clustersize[k] <= 0 ) clustersize[k] = 1;
			inv[k] = 1.0/double(clustersize[k]);//computing inverse now to multiply, than divide later
		}}
		
		{for( int k = 0; k < numk; k++ )
		{
			kseedsl[k] = sigmal[k]*inv[k];
			kseedsa[k] = sigmaa[k]*inv[k];
			kseedsb[k] = sigmab[k]*inv[k];
			kseedsx[k] = sigmax[k]*inv[k];
			kseedsy[k] = sigmay[k]*inv[k];
		}}
	}
}



void SLICO::EnforceLabelConnectivity(
	const int*					labels,//input labels that need to be corrected to remove stray labels
	const int&					width,
	const int&					height,
	int*						nlabels,//new labels
	int&						numlabels,//the number of labels changes in the end if segments are removed
	const int&					K) //the number of superpixels desired by the user
{
//	const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
//	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};

	const int dx4[4] = {-1,  0,  1,  0};
	const int dy4[4] = { 0, -1,  0,  1};

	const int sz = width*height;
	const int SUPSZ = sz/K;
	//nlabels.resize(sz, -1);
	for( int i = 0; i < sz; i++ ) nlabels[i] = -1;
	int label(0);
	int* xvec = new int[sz];
	int* yvec = new int[sz];
	int oindex(0);
	int adjlabel(0);//adjacent label
	for( int j = 0; j < height; j++ )
	{
		for( int k = 0; k < width; k++ )
		{
			if( 0 > nlabels[oindex] )
			{
				nlabels[oindex] = label;
				//--------------------
				// Start a new segment
				//--------------------
				xvec[0] = k;
				yvec[0] = j;
				//-------------------------------------------------------
				// Quickly find an adjacent label for use later if needed
				//-------------------------------------------------------
				{for( int n = 0; n < 4; n++ )
				{
					int x = xvec[0] + dx4[n];
					int y = yvec[0] + dy4[n];
					if( (x >= 0 && x < width) && (y >= 0 && y < height) )
					{
						int nindex = y*width + x;
						if(nlabels[nindex] >= 0) adjlabel = nlabels[nindex];
					}
				}}

				int count(1);
				for( int c = 0; c < count; c++ )
				{
					for( int n = 0; n < 4; n++ )
					{
						int x = xvec[c] + dx4[n];
						int y = yvec[c] + dy4[n];

						if( (x >= 0 && x < width) && (y >= 0 && y < height) )
						{
							int nindex = y*width + x;

							if( 0 > nlabels[nindex] && labels[oindex] == labels[nindex] )
							{
								xvec[count] = x;
								yvec[count] = y;
								nlabels[nindex] = label;
								count++;
							}
						}

					}
				}
				//-------------------------------------------------------
				// If segment size is less then a limit, assign an
				// adjacent label found before, and decrement label count.
				//-------------------------------------------------------
				if(count <= SUPSZ >> 2)
				{
					for( int c = 0; c < count; c++ )
					{
						int ind = yvec[c]*width+xvec[c];
						nlabels[ind] = adjlabel;
					}
					label--;
				}
				label++;
			}
			oindex++;
		}
	}
	numlabels = label;

	if(xvec) delete [] xvec;
	if(yvec) delete [] yvec;
}


void SLICO::PerformSLICO_ForGivenStepSize(
	const unsigned int*			ubuff,
	const int					width,
	const int					height,
	int*						klabels,
	int&						numlabels,
	const int&					STEP,
	const double&				m)
{
	vector<double> kseedsl(0);
	vector<double> kseedsa(0);
	vector<double> kseedsb(0);
	vector<double> kseedsx(0);
	vector<double> kseedsy(0);

	//--------------------------------------------------
	m_width  = width;
	m_height = height;
	int sz = m_width*m_height;
	//klabels.resize( sz, -1 );
	//--------------------------------------------------
	//klabels = new int[sz];
	for( int s = 0; s < sz; s++ ) klabels[s] = -1;
	//--------------------------------------------------
	DoRGBtoLABConversion(ubuff, m_lvec, m_avec, m_bvec);
	//--------------------------------------------------

	bool perturbseeds(true);
	vector<double> edgemag(0);
	if(perturbseeds) DetectLabEdges(m_lvec, m_avec, m_bvec, m_width, m_height, edgemag);
	GetLABXYSeeds_ForGivenStepSize(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, STEP, perturbseeds, edgemag);

	PerformSuperpixelSegmentation_VariableSandM(kseedsl,kseedsa,kseedsb,kseedsx,kseedsy,klabels,STEP,10);
	numlabels = kseedsl.size();

	int* nlabels = new int[sz];
	EnforceLabelConnectivity(klabels, m_width, m_height, nlabels, numlabels, double(sz)/double(STEP*STEP));
	{for(int i = 0; i < sz; i++ ) klabels[i] = nlabels[i];}
	if(nlabels) delete [] nlabels;
}


void SLICO::PerformSLICO_ForGivenK(
	const unsigned int*			ubuff,
	const int					width,
	const int					height,
	int*						klabels,
	int&						numlabels,
	const int&					K,//required number of superpixels
	const double&				m)//weight given to spatial distance
{
	vector<double> kseedsl(0);
	vector<double> kseedsa(0);
	vector<double> kseedsb(0);
	vector<double> kseedsx(0);
	vector<double> kseedsy(0);

	//--------------------------------------------------
	m_width  = width;
	m_height = height;
	int sz = m_width*m_height;
	//--------------------------------------------------
	//if(0 == klabels) klabels = new int[sz];
	for( int s = 0; s < sz; s++ ) klabels[s] = -1;
	//--------------------------------------------------
	if(1)//LAB
	{
		DoRGBtoLABConversion(ubuff, m_lvec, m_avec, m_bvec);
	}
	else//RGB
	{
		m_lvec = new double[sz]; m_avec = new double[sz]; m_bvec = new double[sz];
		for( int i = 0; i < sz; i++ )
		{
			m_lvec[i] = ubuff[i] >> 16 & 0xff;
			m_avec[i] = ubuff[i] >>  8 & 0xff;
			m_bvec[i] = ubuff[i]       & 0xff;
		}
	}
	//--------------------------------------------------

	bool perturbseeds(true);
	vector<double> edgemag(0);
	if(perturbseeds) DetectLabEdges(m_lvec, m_avec, m_bvec, m_width, m_height, edgemag);
	GetLABXYSeeds_ForGivenK(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, K, perturbseeds, edgemag);

	int STEP = sqrt(double(sz)/double(K)) + 2.0;//adding a small value in the even the STEP size is too small.
	//PerformSuperpixelSLIC(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, klabels, STEP, edgemag, m);
	PerformSuperpixelSegmentation_VariableSandM(kseedsl,kseedsa,kseedsb,kseedsx,kseedsy,klabels,STEP,10);
	numlabels = kseedsl.size();

	int* nlabels = new int[sz];
	EnforceLabelConnectivity(klabels, m_width, m_height, nlabels, numlabels, K);
	{for(int i = 0; i < sz; i++ ) klabels[i] = nlabels[i];}
	if(nlabels) delete [] nlabels;
}



void SLICO::GetArcAndCenterOfSuperpixels( const Mat& img, vector<int>& klabels, int& numlabels, vector<vector<int> >& arcs, vector<Vec6d>& centers, vector<vector<int> >& spLabels)	
{


	int width = img.cols;
	int height = img.rows;

	////Arcs
	//vector<hash_set<int> > sets(numlabels);
	//for(int r = 0; r < height; r+=2)
	//{
	//	for(int c = 0; c < width; c+=2)
	//	{
	//		int index = r * width + c;

	//		for(int k = 0; k < 4; k++)
	//		{
	//			int newr = r + dy4[k];
	//			int newc = c + dx4[k];
	//			if((height > newr && newr >= 0) && (width > newc && newc >= 0))
	//			{
	//				int newIndex = newr * width + newc;
	//				if(klabels[index] != klabels[newIndex])
	//				{
	//					int lmin = std::min(klabels[index], klabels[newIndex]);
	//					int lmax = std::max(klabels[index], klabels[newIndex]);
	//					sets[lmin].insert(lmax);
	//					sets[lmax].insert(lmin);
	//				}
	//			}
	//		}
	//	}
	//}

	//if(arcs.size() != numlabels) arcs.resize(numlabels);
	//hash_set<int>::iterator iter;
	//for(int i = 0; i < numlabels; i++)
	//{
	//	arcs[i].clear();
	//	for(iter = sets[i].begin(); iter != sets[i].end(); iter++)
	//	{
	//		arcs[i].push_back(*iter);
	//	}
	//}


	//Centers
	vector<int> labelCount(numlabels,0);

	vector<double> sumA(numlabels, 0);
	vector<double> sumR(numlabels, 0);
	vector<double> sumG(numlabels, 0);
	vector<double> sumB(numlabels, 0);
	vector<double> sumX(numlabels, 0);
	vector<double> sumY(numlabels, 0);


	spLabels.resize(numlabels);

	Point p;
	int index = 0;
    for( p.y = 0; p.y < img.rows; p.y++ )
    {
        for( p.x = 0; p.x < img.cols; p.x++ )
        {
			int index = p.y * width + p.x;
			int label = klabels[index];
			Vec3d color = img.at<Vec3b>(p);

			labelCount[label]++;

			sumR[label] += color[2];
			sumG[label] += color[1];
			sumB[label] += color[0];
			sumY[label] += p.y;
			sumX[label] += p.x;

			spLabels[label].push_back(index);
		}
	}

	//int maxCount = -1;
	//int minCount = INT_MAX;
	Vec6d v6;
	v6[0] = 0, v6[1] = 0, v6[2] = 0, v6[3] = 0, v6[4] = 0, v6[5] = 0;
	if(centers.size() != numlabels) centers.resize(numlabels, v6);
	for(int i = 0; i < numlabels; i++)
	{
		sumR[i] /= labelCount[i];
		sumG[i] /= labelCount[i];
		sumB[i] /= labelCount[i];
		sumX[i] /= labelCount[i];
		sumY[i] /= labelCount[i];

		int centerR = ((uint)(sumR[i]+0.5)) & 0xFF;
		int centerG = ((uint)(sumG[i]+0.5)) & 0xFF;
		int centerB = ((uint)(sumB[i]+0.5)) & 0xFF;

		centers[i][0] = centerB;
		centers[i][1] = centerG;
		centers[i][2] = centerR;
		centers[i][3] = labelCount[i];	//超像素大小（包含像素个数）
		//if(labelCount[i] > maxCount) maxCount = labelCount[i];
		//if(labelCount[i] < minCount) minCount = labelCount[i];
		//centers[i][3] = centerA;
		centers[i][4] = sumX[i];
		centers[i][5] = sumY[i];
	}

	//Arcs
	if(arcs.size() != numlabels) arcs.resize(numlabels);
	//cout<<"##########################max="<<maxCount<<"; min="<<minCount<<endl;

	for(int i = 0; i < numlabels; i++)
	{
		for(int j = 0; j < numlabels; j++)
		{
			if(i == j) continue;
			if(abs(centers[j][4]-centers[i][4]) < m_size && abs(centers[j][5]-centers[i][5]) < m_size)
			{
				arcs[i].push_back(j);
			}

		}
	}


}


void SLICO::DrawAverageColor(Mat &mat, vector<int>& klabels, vector<Vec6d>& centers)
{
		
	for(int r = 0; r < mat.rows; r++)
	{
		for(int c = 0; c < mat.cols; c++)
		{
			int pinIndex = r * mat.cols + c;
			int spIndex = klabels[pinIndex];
			Vec6d v6 = centers[spIndex];
			Vec3d v3;
			v3[0] = v6[0];
			v3[1] = v6[1];
			v3[2] = v6[2];
			mat.at<Vec3b>(r,c) = v3;
		}
	}
}


//=================================================================
//对Mat图像进行分割
//=================================================================
void SLICO::DoSuperpixelSegmentation_ForGivenMat( const Mat& img, vector<int>& klabels, int& numlabels)							
{
	int width = img.cols;
	int height = img.rows;
	int size = width * height;

	uint* ubuff = (uint*)malloc(sizeof(uint)*size);

	Point p;
	int index = 0;
    for( p.y = 0; p.y < img.rows; p.y++ )
    {
        for( p.x = 0; p.x < img.cols; p.x++ )
        {
			Vec3d color = img.at<Vec3b>(p);
			uint b = color[0];
			uint g = color[1];
			uint r = color[2];
			uint rgb = (r<<16) | (g<<8) | (b);
			int index = p.y * img.cols + p.x;
			ubuff[index] = rgb;
		}
	}


	//int nr = img.rows;
	//int nc = img.cols;
	//int channel = img.channels();
	//if(img.isContinuous())
	//{
	//	nr = 1;
	//	nc = nc * img.rows * channel;
	//}
 //   for(int y = 0; y < nr; y++ )
 //   {
	//	const uchar* data = img.ptr<uchar>(y);
 //       for(int x = 0; x < nc; x += channel )
 //       {
	//		uint b = *(data+x);
	//		uint g = *(data+x+1);
	//		uint r = *(data+x+2);

	//		uint rgb = (r<<16) | (g<<8) | (b);
	//		int index = y * (nc/channel) + x/channel;
	//		ubuff[index] = rgb;
	//	}
	//}





	int* _klabels = (int*)malloc(sizeof(int)*size);
	numlabels = 0;
	int STEP = 10;  // 10 =1536个 // 16=600个
	double compactness = 20.0;

	PerformSLICO_ForGivenStepSize(ubuff, width, height, _klabels, numlabels, STEP, compactness);
	delete ubuff;

	klabels.resize(size);
	for(int i = 0; i < size; i++)
		klabels[i] = _klabels[i];
	delete _klabels;
}


