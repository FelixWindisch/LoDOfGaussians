

class HierarchyCut
{
    public:
        static int getHierarchyCut(
	    	int N, 
	    	float target_size, 
	    	int *nodes, 
	    	float *positions, 
	    	float *scales, 
	    	float *viewpoint, 
	    	float* zdir, 
	    	int *render_indices);
};