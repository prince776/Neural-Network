package neuralNetwork;

public class NeuralNetwork {
	
	private int input_nodes,hidden_nodes,output_nodes;
	
	private Matrix weights_ih,weights_ho;
	private Matrix bias_h , bias_o;
	private float learning_rate;
	/**
	 * 
	 * @param input_nodes
	 * @param hidden_nodes
	 * @param output_nodes
	 */
	public NeuralNetwork(int input_nodes, int hidden_nodes, int output_nodes) {
		this.input_nodes = input_nodes;
		this.hidden_nodes = hidden_nodes;
		this.output_nodes = output_nodes;
		
		this.weights_ih = new Matrix(hidden_nodes, input_nodes);
		this.weights_ho = new Matrix(output_nodes, hidden_nodes);
		this.weights_ih.randomize();
		this.weights_ho.randomize();
		
		this.bias_h = new Matrix(hidden_nodes, 1);
		this.bias_o = new Matrix(output_nodes, 1);
		bias_h.randomize();
		bias_o.randomize();
		this.learning_rate=0.1f;
	}
	/**
	 * 
	 * @param nn
	 */
	public NeuralNetwork(NeuralNetwork nn) {
		this.input_nodes = nn.input_nodes;
		this.hidden_nodes = nn.hidden_nodes;
		this.output_nodes = nn.output_nodes;
		
		this.weights_ih = nn.weights_ih.copy();
		this.weights_ho = nn.weights_ho.copy();
		
		this.bias_h = nn.bias_h.copy();
		this.bias_o = nn.bias_o.copy();
		this.learning_rate=nn.learning_rate;
	}
	/**
	 * 
	 * @param input_array
	 * @return
	 */
	public float[] predict(float[] input_array){
		
		Matrix inputs = Matrix.fromArray(input_array);
		
		//CALCULATE OUTPUT OF HIDDEN NODES = INPUT FOR OUPUT PERCEPTRONS
		Matrix hidden = Matrix.matrixMultiply(weights_ih, inputs);
		hidden.add(this.bias_h);
		hidden.sigmoid();	//activation function
		
		//CALCULATE OUTPUT
		Matrix output = Matrix.matrixMultiply(this.weights_ho, hidden);
		output.add(bias_o);
		output.sigmoid(); //activation function
		
		return output.toArray();
		
	}
	/**
	 * 
	 * @param input_array
	 * @param targets
	 */
	public void train(float[] input_array , float[] targets){
		
		Matrix inputs = Matrix.fromArray(input_array);
		
		//CALCULATE OUTPUT OF HIDDEN NODES = INPUT FOR OUPUT PERCEPTRONS
		Matrix hidden = Matrix.matrixMultiply(weights_ih, inputs);
		hidden.add(this.bias_h);
		hidden.sigmoid();	//activation function
		
		//CALCULATE OUTPUT
		Matrix outputs = Matrix.matrixMultiply(this.weights_ho, hidden);
		outputs.add(bias_o);
		outputs.sigmoid(); //activation function
		
		Matrix target = Matrix.fromArray(targets);
		
		//CALCULATE THE ERROR (= TARGETS - OUTPUTS)
		Matrix output_errors = Matrix.subtract(target, outputs);
		
		
		//TUNE THE WEIGHTS delta(W) = learningrate*(dsigmoid(outputs)*output_erros)x input , input in case of outermost layer is hidden output
		//in perceptron we had delta(W) = lr * error*input(corresponding input)
		//here we have delta(W) = lr*(error*dsigmoid)x input(corresponding input) * -> scalar matrix product and x->cross product
		
		//CALCULATE GRADIENT( = dsigmoid*error*lr)
		Matrix gradients = Matrix.dsigmoid(outputs);
		gradients.scalarMultiply(output_errors);
		gradients.multiply(learning_rate);//at this stage graident = bias, as input value for it =1;
		
		
		//CACLULATE delat(W) for h->o layer
		Matrix hidden_T =Matrix.transpose(hidden);
		Matrix weights_ho_delta = Matrix.matrixMultiply(gradients, hidden_T);
		//TUNE WEIGHT
		this.weights_ho.add(weights_ho_delta);
		bias_o.add(gradients);

		//FOR MUTI LAYERED NEURAL NETWORK(more than 1 hidden layers) we'll have this in loop
		//This is the back propagation crap
		Matrix who_t =Matrix.transpose(this.weights_ho);//who_t -> weight_hiddenToOutputTranspose
		Matrix hidden_error  = Matrix.matrixMultiply(who_t, output_errors);
		
		//CALCULATE HIDDEN GRADIENT
		Matrix hidden_gradient = Matrix.dsigmoid(hidden);
		hidden_gradient.scalarMultiply(hidden_error);
		hidden_gradient.multiply(learning_rate);
		//CACLULATE delat(W) for i->h layer
		Matrix inputs_T=Matrix.transpose(inputs);
		Matrix weight_ih_deltas =  Matrix.matrixMultiply(hidden_gradient, inputs_T);
		this.weights_ih.add(weight_ih_deltas);
		bias_h.add(hidden_gradient);

	}
	
	public NeuralNetwork copy(){
		NeuralNetwork result = new NeuralNetwork(this);
		return result;
	}
	
	public static NeuralNetwork crossover(NeuralNetwork nn1,NeuralNetwork nn2){
		NeuralNetwork nn = new NeuralNetwork(nn1);
		nn.weights_ih = Matrix.crossover(nn1.weights_ih,nn2.weights_ih);
		nn.weights_ho = Matrix.crossover(nn1.weights_ho,nn2.weights_ho);
		nn.bias_h = Matrix.crossover(nn1.bias_h,nn2.bias_h);
		nn.bias_o = Matrix.crossover(nn1.bias_o,nn2.bias_o);
		return nn;
	}
	
	public static NeuralNetwork fixedCrossover(NeuralNetwork nn1,NeuralNetwork nn2){
		NeuralNetwork nn = new NeuralNetwork(nn1);
		nn.weights_ih = Matrix.fixedCrossover(nn1.weights_ih,nn2.weights_ih);
		nn.weights_ho = Matrix.fixedCrossover(nn1.weights_ho,nn2.weights_ho);
		nn.bias_h = Matrix.fixedCrossover(nn1.bias_h,nn2.bias_h);
		nn.bias_o = Matrix.fixedCrossover(nn1.bias_o,nn2.bias_o);
		return nn;
	}

	
	/**
	 * 
	 * @param mutationRate
	 * @param factor
	 */
	public void mutate(float mutationRate,float factor){
		weights_ih.mutate(mutationRate,factor);
		weights_ho.mutate(mutationRate,factor);
		bias_h.mutate(mutationRate,factor);
		bias_o.mutate(mutationRate,factor);

	}
	
	public float getLearning_rate() {
		return learning_rate;
	}

	public void setLearning_rate(float learning_rate) {
		this.learning_rate = learning_rate;
	}
	
	
	
}
