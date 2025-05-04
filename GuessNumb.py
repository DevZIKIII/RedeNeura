import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self):
        # Initialize weights and bias to None
        self.w1 = None
        self.w2 = None
        self.bias = None
        
    def treinamento(self, entradas, saidas, taxaAprendizado=0.1, epocas=1000):
        """
        Treina o perceptron com conjunto de dados
        
        Parâmetros:
        - entradas: array de entradas de treinamento
        - saidas: array de saídas esperadas
        - taxaAprendizado: taxa de aprendizado (default=0.1)
        - epocas: número de épocas de treinamento (default=1000)
        
        Retorna:
        - w1, w2, bias: pesos e viés treinados
        """
        self.entradas = entradas
        self.saidas = saidas
        self.taxaAprendizado = taxaAprendizado
        self.epocas = epocas
        
        # Inicialização dos pesos e bias aleatoriamente
        self.w1 = np.random.uniform(-1, 1)
        self.w2 = np.random.uniform(-1, 1)
        self.bias = np.random.uniform(-1, 1)
        
        # Histórico de erros para visualização
        self.historico_erros = []
        
        # Definir erro mínimo para critério de parada
        erro_minimo = 0.001
        erro_epoca = float('inf')
        
        # Garantir que o treinamento não exceda o número máximo de épocas
        for i in range(self.epocas):
            erro_epoca = 0
            
            for j in range(len(self.entradas)):
                # Aplicar função de ativação sigmoid
                net = (self.entradas[j][0] * self.w1) + (self.entradas[j][1] * self.w2) + self.bias
                sigmoid = 1 / (1 + np.exp(-net))
                
                # Cálculo do erro
                erro = self.saidas[j][0] - sigmoid
                erro_epoca += abs(erro)
                
                # Atualização dos pesos
                self.w1 = self.w1 + (self.taxaAprendizado * erro * self.entradas[j][0])
                self.w2 = self.w2 + (self.taxaAprendizado * erro * self.entradas[j][1])
                self.bias = self.bias + (self.taxaAprendizado * erro)
            
            # Calcula o erro médio da época
            erro_medio = erro_epoca / len(self.entradas)
            self.historico_erros.append(erro_medio)
            
            # Mostrar progresso a cada 100 épocas
            if (i+1) % 100 == 0:
                print(f"Época {i+1}/{self.epocas} - Erro médio: {erro_medio:.6f}")
            
            # Critério de parada antecipada se o erro for muito pequeno
            if erro_medio < erro_minimo:
                print(f"Treinamento convergiu na época {i+1} com erro {erro_medio:.6f}")
                break
                
        # Caso tenha atingido o número máximo de épocas
        if i == self.epocas - 1:
            print(f"Treinamento concluído após {self.epocas} épocas com erro final {erro_medio:.6f}")
                
        return self.w1, self.w2, self.bias
    
    def predicao(self, entrada):
        """
        Realiza a predição para uma nova entrada
        
        Parâmetros:
        - entrada: array com valores de entrada
        
        Retorna:
        - resultado da predição (0 ou 1)
        """
        if self.w1 is None or self.w2 is None or self.bias is None:
            raise ValueError("O perceptron precisa ser treinado antes de fazer predições!")
            
        net = (entrada[0] * self.w1) + (entrada[1] * self.w2) + self.bias
        sigmoid = 1 / (1 + np.exp(-net))
        
        # Retorna 1 se sigmoid > 0.5, caso contrário 0
        return 1 if sigmoid > 0.5 else 0
    
    def plot_treinamento(self):
        """Plota o histórico de erros durante o treinamento"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.historico_erros)
        plt.title('Erro ao longo do treinamento')
        plt.xlabel('Épocas')
        plt.ylabel('Erro médio')
        plt.grid(True)
        plt.show()
        
    def plot_decisao(self, entradas, saidas):
        """Plota a fronteira de decisão e os dados de treinamento"""
        plt.figure(figsize=(10, 6))
        
        # Plotar pontos de treinamento
        for i in range(len(entradas)):
            marker = 'o' if saidas[i][0] == 1 else 'x'
            color = 'blue' if saidas[i][0] == 1 else 'red'
            plt.scatter(entradas[i][0], entradas[i][1], marker=marker, color=color, s=100)
        
        # Criar pontos para desenhar a linha de decisão
        x_min, x_max = np.min(entradas[:, 0]) - 1, np.max(entradas[:, 0]) + 1
        y_min, y_max = np.min(entradas[:, 1]) - 1, np.max(entradas[:, 1]) + 1
        
        # Equação da linha: w1*x + w2*y + bias = 0 => y = (-w1*x - bias) / w2
        xx = np.linspace(x_min, x_max, 100)
        yy = (-self.w1 * xx - self.bias) / self.w2
        
        plt.plot(xx, yy, 'k-', label='Fronteira de decisão')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xlabel('Valor do chute')
        plt.ylabel('Diferença do chute anterior')
        plt.title('Fronteira de Decisão do Perceptron')
        plt.legend()
        plt.grid(True)
        plt.show()


class GuessNumberGame:
    def __init__(self, min_num=1, max_num=100):
        """
        Inicializa o jogo de adivinhar números
        
        Parâmetros:
        - min_num: valor mínimo para o número secreto
        - max_num: valor máximo para o número secreto
        """
        self.min_num = min_num
        self.max_num = max_num
        self.reset_game()
        self.perceptron = Perceptron()
        self.trained = False
        
    def reset_game(self):
        """Reinicia o jogo com um novo número secreto"""
        self.secret_number = np.random.randint(self.min_num, self.max_num + 1)
        self.attempts = []
        self.current_min = self.min_num
        self.current_max = self.max_num
        return self.secret_number
        
    def generate_training_data(self, num_examples=1000):
        """
        Gera dados sintéticos para treinar o perceptron
        
        Parâmetros:
        - num_examples: número de exemplos a serem gerados
        
        Retorna:
        - X: matriz de características (chute atual, diferença do chute anterior)
        - y: vetor de direções (1: maior, 0: menor)
        """
        X = np.zeros((num_examples, 2))
        y = np.zeros((num_examples, 1))
        
        for i in range(num_examples):
            # Gerar um número aleatório como número secreto
            secret = np.random.randint(self.min_num, self.max_num + 1)
            
            # Gerar um chute aleatório
            guess = np.random.randint(self.min_num, self.max_num + 1)
            
            # Gerar uma diferença aleatória do chute anterior (normalizada)
            prev_diff = np.random.uniform(-1, 1)
            
            # Normalizar o chute para o intervalo [0, 1]
            normalized_guess = (guess - self.min_num) / (self.max_num - self.min_num)
            
            # Definir a direção correta (maior=1, menor=0)
            direction = 1 if guess < secret else 0
            
            X[i] = [normalized_guess, prev_diff]
            y[i] = [direction]
            
        return X, y
    
    def train_perceptron(self, num_examples=1000, taxa_aprendizado=0.1, epocas=1000):
        """Treina o perceptron com dados sintéticos"""
        print(f"Gerando {num_examples} exemplos de treinamento...")
        X, y = self.generate_training_data(num_examples)
        
        print(f"Iniciando treinamento do perceptron com {epocas} épocas...")
        self.perceptron.treinamento(X, y, taxa_aprendizado, epocas)
        self.trained = True
        
        print("Treinamento concluído!")
        
        # Perguntar ao usuário se deseja ver os gráficos
        visualizar = input("Deseja visualizar os gráficos de treinamento? (s/n): ").lower()
        if visualizar == 's' or visualizar == 'sim':
            # Plotar o processo de treinamento
            self.perceptron.plot_treinamento()
            self.perceptron.plot_decisao(X, y)
        
    def get_ai_guess(self):
        """Obtém um chute da IA com base no perceptron treinado"""
        if not self.trained:
            raise ValueError("O perceptron precisa ser treinado antes de fazer chutes!")
            
        if not self.attempts:
            # Primeiro chute: usar o ponto médio
            return (self.current_min + self.current_max) // 2
        
        last_guess = self.attempts[-1]
        
        # Normalizar o último chute
        norm_guess = (last_guess - self.min_num) / (self.max_num - self.min_num)
        
        # Calcular a diferença normalizada do penúltimo chute (se existir)
        if len(self.attempts) >= 2:
            prev_diff = (self.attempts[-1] - self.attempts[-2]) / (self.max_num - self.min_num)
        else:
            prev_diff = 0
            
        # Obter a direção prevista pelo perceptron
        direction = self.perceptron.predicao([norm_guess, prev_diff])
        
        if direction == 1:  # Precisa chutar maior
            new_guess = (last_guess + self.current_max) // 2
            self.current_min = last_guess + 1
        else:  # Precisa chutar menor
            new_guess = (self.current_min + last_guess) // 2
            self.current_max = last_guess - 1
            
        # Garantir que o chute esteja dentro dos limites
        new_guess = max(self.current_min, min(new_guess, self.current_max))
        
        return new_guess
    
    def make_guess(self, guess):
        """
        Processa um chute e retorna o resultado
        
        Parâmetros:
        - guess: valor do chute
        
        Retorna:
        - Resultado: "Correto!", "Muito baixo" ou "Muito alto"
        """
        self.attempts.append(guess)
        
        if guess == self.secret_number:
            return "Correto!"
        elif guess < self.secret_number:
            self.current_min = max(self.current_min, guess + 1)
            return "Muito baixo"
        else:
            self.current_max = min(self.current_max, guess - 1)
            return "Muito alto"
            
    def play_human(self):
        """Interface para o usuário humano jogar"""
        self.reset_game()
        print(f"Estou pensando em um número entre {self.min_num} e {self.max_num}.")
        
        while True:
            try:
                guess = int(input("Seu chute: "))
                result = self.make_guess(guess)
                print(result)
                
                if result == "Correto!":
                    print(f"Parabéns! Você acertou em {len(self.attempts)} tentativas.")
                    break
            except ValueError:
                print("Por favor, digite um número válido.")
    
    def play_ai(self, max_attempts=10):
        """Faz a IA jogar automaticamente contra o computador"""
        if not self.trained:
            print("Treinando o perceptron primeiro...")
            self.train_perceptron()
            
        self.reset_game()
        print(f"A IA está tentando adivinhar o número secreto: {self.secret_number}")
        
        for i in range(max_attempts):
            guess = self.get_ai_guess()
            result = self.make_guess(guess)
            print(f"Tentativa {i+1}: IA chutou {guess} - {result}")
            
            if result == "Correto!":
                print(f"A IA acertou em {len(self.attempts)} tentativas!")
                return True
                
        print(f"A IA não conseguiu adivinhar o número {self.secret_number} em {max_attempts} tentativas.")
        return False
                
    def play_ai_guessing(self):
        """Modo de jogo onde a IA tenta adivinhar o número que o usuário está pensando"""
        if not self.trained:
            print("Treinando o perceptron primeiro...")
            self.train_perceptron()
            
        print("\nPense em um número entre", self.min_num, "e", self.max_num)
        print("A IA tentará adivinhar seu número. Responda se o número que você pensou é maior ou menor que o chute da IA.")
        
        # Resetar limites
        self.current_min = self.min_num
        self.current_max = self.max_num
        self.attempts = []
        
        tentativa = 1
        while True:
            # Obter chute da IA
            guess = self.get_ai_guess()
            print(f"\nTentativa {tentativa}: A IA chuta {guess}")
            
            # Obter feedback do usuário com instruções mais claras
            while True:
                feedback = input(f"O número que você pensou é MAIOR que {guess} (m), MENOR que {guess} (n) ou IGUAL a {guess} (c)? ").lower()
                if feedback in ['m', 'n', 'c']:
                    break
                print("Por favor, digite 'm' se seu número é maior, 'n' se seu número é menor ou 'c' se é correto.")
            
            # Armazenar o chute
            self.attempts.append(guess)
            
            # Processar feedback - CORREÇÃO NA LÓGICA AQUI:
            if feedback == 'c':
                print(f"Ótimo! A IA acertou o seu número em {tentativa} tentativas!")
                break
            elif feedback == 'm':  # Número secreto é MAIOR que o chute
                # Atualizar o limite mínimo pois devemos procurar valores maiores
                self.current_min = max(self.current_min, guess + 1)
            elif feedback == 'n':  # Número secreto é MENOR que o chute
                # Atualizar o limite máximo pois devemos procurar valores menores
                self.current_max = min(self.current_max, guess - 1)
            
            # Verificar se ainda há números possíveis
            if self.current_min > self.current_max:
                print("Hmm... Parece que há uma inconsistência nas suas respostas.")
                print(f"Não há números possíveis entre {self.current_min} e {self.current_max}.")
                print("Verifique se você está respondendo corretamente: ")
                print("- 'm' se seu número é MAIOR que o chute da IA")
                print("- 'n' se seu número é MENOR que o chute da IA")
                print("- 'c' se seu número é IGUAL ao chute da IA")
                break
                
            tentativa += 1


# Demonstração do jogo
if __name__ == "__main__":
    print("Bem-vindo ao Jogo de Adivinhar Números com Perceptron!")
    game = GuessNumberGame(min_num=1, max_num=100)
    
    # Treinar o perceptron (com parâmetros mais razoáveis)
    print("Treinando o perceptron...")
    game.train_perceptron(num_examples=1000, taxa_aprendizado=0.1, epocas=500)
    
    while True:
        print("\nEscolha uma opção:")
        print("1. Você tenta adivinhar o número do computador")
        print("2. O computador tenta adivinhar seu número")
        print("3. Simular a IA adivinhando números aleatórios")
        print("4. Sair")
        
        choice = input("Opção: ")
        
        if choice == "1":
            game.play_human()
        elif choice == "2":
            game.play_ai_guessing()
        elif choice == "3":
            num_games = int(input("Quantos jogos a IA deve jogar? "))
            wins = 0
            total_attempts = 0
            
            for i in range(num_games):
                print(f"\nJogo {i+1}/{num_games}")
                secret = game.reset_game()
                if game.play_ai(max_attempts=15):
                    wins += 1
                    total_attempts += len(game.attempts)
                    
            success_rate = (wins / num_games) * 100
            avg_attempts = total_attempts / wins if wins > 0 else 0
            
            print(f"\nResultados da simulação:")
            print(f"Taxa de sucesso: {success_rate:.2f}%")
            print(f"Número médio de tentativas: {avg_attempts:.2f}")
        elif choice == "4":
            print("Obrigado por jogar!")
            break
        else:
            print("Opção inválida. Tente novamente.")