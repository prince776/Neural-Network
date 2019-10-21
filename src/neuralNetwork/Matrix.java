package neuralNetwork;


public class Matrix {
	
	private int rows,cols;
	private float data[][]; 
	
	public Matrix(int rows,int cols){
		this.rows=rows;
		this.cols= cols;
		this.data = new float[rows][cols];
		for(int i=0;i<rows;i++){
			for(int j=0;j<cols;j++){
				this.data[i][j] = 0;
			}
		}
	}
	
	public static Matrix crossover(Matrix m1,Matrix m2){
		Matrix result = new Matrix(m1.rows, m1.cols);
		for(int i=0;i<result.rows;i++){
			int colMid = (int)Math.random()*m1.cols;
			for(int j=0;j<result.cols;j++){
				if(j<colMid){
					result.data[i][j] = m1.data[i][j];
				}else{
					result.data[i][j] = m2.data[i][j];
				}
			}
		}		
		return result;
	}
	
	public static Matrix fixedCrossover(Matrix m1,Matrix m2){
		Matrix result = new Matrix(m1.rows, m1.cols);
		for(int i=0;i<result.rows;i++){
			int colMid = result.cols/2;
			for(int j=0;j<result.cols;j++){
				if(j<colMid){
					result.data[i][j] = m1.data[i][j];
				}else{
					result.data[i][j] = m2.data[i][j];
				}
			}
		}		
		return result;
	}

	
	public Matrix copy(){
		Matrix m  = new Matrix(rows, cols);
		for(int i=0;i<rows;i++){
			for(int j=0;j<cols;j++){
				m.data[i][j] = this.data[i][j];
			}
		}
		return m;
	}
	
	public void randomize(){
		for(int i=0;i<rows;i++){
			for(int j=0;j<cols;j++){
				this.data[i][j] =( (float) (Math.random()*2)-1f);
			}
		}
	}
	
	public static Matrix transpose(Matrix m){
		Matrix result = new Matrix(m.cols, m.rows);
		for(int i=0;i<m.rows;i++){
			for(int j=0;j<m.cols;j++){
				result.data[j][i] =	m.data[i][j];
			}
		}
		return result;
	}
	
	public float[] toArray(){
		float[] arr = new float[rows*cols];
		for(int i=0;i<rows;i++){
			for(int j=0;j<cols;j++){
				arr[i*cols+j] = this.data[i][j];
			}
		}
		return arr;
	}
	
	//scalars operations
	public void multiply(float n){
		for(int i=0;i<rows;i++){
			for(int j=0;j<cols;j++){
				this.data[i][j] *= n;
			}
		}
	}
	
	public void add(float n){
		for(int i=0;i<rows;i++){
			for(int j=0;j<cols;j++){
				this.data[i][j] += n;
			}
		}
	}
	
	//matrix operations
	public void add(Matrix m){
		for(int i=0;i<rows;i++){
			for(int j=0;j<cols;j++){
				this.data[i][j] += m.data[i][j];
			}
		}
	}
	
	public void subtract(Matrix m){
		for(int i=0;i<rows;i++){
			for(int j=0;j<cols;j++){
				this.data[i][j] -= m.data[i][j];
			}
		}
	}
	
	public void scalarMultiply(Matrix m){
		for(int i=0;i<rows;i++){
			for(int j=0;j<cols;j++){
				this.data[i][j] *= m.data[i][j];
			}
		}
	}
	
	public void matrixMultiply(Matrix m){
		
		if(this.cols!=m.rows){
			System.out.println("Can't cross multiply given matrix!");
			return ;
		}
		Matrix result = new Matrix(this.rows,m.cols);
		float[][] a = this.data;
		float[][] b = m.data;
		for(int i=0;i<result.rows;i++){
			for(int j=0;j<result.cols;j++){
				float sum=0;
				for(int k=0;k<this.cols;k++){
					sum+= a[i][k] * b[k][j];
				}
				result.data[i][j]=sum;
			}
		}
		for(int i=0;i<rows;i++){
			for(int j=0;j<cols;j++){
				this.data[i][j] += result.data[i][j];
			}
		}
		
	}
	
	public  void dsigmoid(){
		for(int i=0;i<this.rows;i++){
			for(int j=0;j<this.cols;j++){
				this.data[i][j] = this.data[i][j]*(1-this.data[i][j]);
			}
		}
	}
	
	//static methods for scalar and matrix operations
	
	public static Matrix scalarMultiply(Matrix a,Matrix b){
		Matrix result = new Matrix(a.rows,a.cols);
		for(int i=0;i<result.rows;i++){
			for(int j=0;j<result.cols;j++){
				result.data[i][j] = a.data[i][j] * b.data[i][j];
			}
		}
		return result;
	}
	

	public static Matrix add(Matrix a,Matrix b){
		Matrix result = new Matrix(a.rows,a.cols);
		for(int i=0;i<result.rows;i++){
			for(int j=0;j<result.cols;j++){
				result.data[i][j] = a.data[i][j] + b.data[i][j];
			}
		}
		return result;
	}
	
	public static Matrix subtract(Matrix a,Matrix b){
		Matrix result = new Matrix(a.rows,a.cols);
		for(int i=0;i<result.rows;i++){
			for(int j=0;j<result.cols;j++){
				result.data[i][j] = a.data[i][j] - b.data[i][j];
			}
		}
		return result;
	}
	
	public void sigmoid(){
		for(int i=0;i<this.rows;i++){
			for(int j=0;j<this.cols;j++){
				this.data[i][j] = sigmoid(this.data[i][j]);
			}
		}
	}
	
	public static Matrix matrixMultiply(Matrix a,Matrix b){
		
		if(a.cols!=b.rows){
			return null;
		}
		Matrix result = new Matrix(a.rows,b.cols);
		for(int i=0;i<result.rows;i++){
			for(int j=0;j<result.cols;j++){
				float sum=0;
				for(int k=0;k<a.cols;k++){
					sum+= a.data[i][k] * b.data[k][j];
				}
				result.data[i][j]=sum;
			}
		}
		return result;
	}
	
	public static Matrix sigmoid(Matrix a){
		Matrix m=new Matrix(a.rows,a.cols);
		for(int i=0;i<a.rows;i++){
			for(int j=0;j<a.cols;j++){
				m.data[i][j] = sigmoid(a.data[i][j]);
			}
		}
		return m;
	}
	
	public static Matrix dsigmoid(Matrix a){
		Matrix m=new Matrix(a.rows,a.cols);
		for(int i=0;i<a.rows;i++){
			for(int j=0;j<a.cols;j++){
				m.data[i][j] = a.data[i][j]*(1-a.data[i][j]);
			}
		}
		return m;
	}
	
	public static Matrix fromArray(float[] arr){
		Matrix result = new Matrix(arr.length, 1);
		for(int i=0;i<arr.length;i++){
			result.data[i][0] =arr[i];
		}
		return result;
	}
	
	public static float sigmoid (float x){
		return (float)(1f/(1+Math.exp(-x)));
	}



	public int getRows() {
		return rows;
	}



	public int getCols() {
		return cols;
	}



	public float[][] getData() {
		return data;
	}
	
	public void mutate(float mutationRate,float factor){
		for(int i=0;i<rows;i++){
			float x = (float) Math.random();
			for(int j=0;j<cols;j++){
				if(x<mutationRate){
					this.data[i][j] += ((float)(Math.random()*2-1)*factor);
				}
			}
		}
	}
	
}
