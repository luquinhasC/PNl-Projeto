using ForwardDiff;
using LinearAlgebra;

## Em Km ##
u_1 = [ 3.718 ,-1.847] ##Poli
u_2 = [ 2.177 , 2.289] ##Agrarias
u_3 = [ 0.861 , 0.728] ##Prédio Histórico
u_4 = [ 0.616 ,-0.405] ##Rebouças
u_5 = [-1.837 , -1.997] ##Puc
λ_1 = 4
λ_2 = λ_3 = 2
λ_4 = λ_5 = 1
U = [u_1,u_2,u_3,u_4,u_5]
#U = [u_1,u_2,u_3]
λ = [λ_1,λ_2,λ_3,λ_4,λ_5]
function f(x)
     x_1 = x[1]
     x_2 = x[2]
     y = 0
     for i in range(1, stop = size(U,1))
        y = y + λ[i]*sqrt((x_1 - U[i][1])^2 + (x_2 - U[i][2])^2 ) 
     end
    return y
end

P_i = [-2,-2]


function Cauchy(x)
    k = x;
    η = 0.1;
    γ = 0.5;
    iter = 0;
    ∇f = ForwardDiff.gradient(f,k)
     
    while norm(∇f) > 1e-6
        if iter >= 1000
            error("Muitas iterações")
        end
        t = 1;
        ∇f = ForwardDiff.gradient(f,k)
        d = (-1)*∇f #Direção de descida
        while (f(k+t*d) > f(k) + η*t*dot(∇f,d)) ##Busca de armijo
            t = γ*t
         end
        k = k + t*d  
        iter +=1
    end
    println("O Método de Cauchy foi aplicado em ", iter, " Iterações")
    println("A norma do gradiente da função no ponto encontrado é ",norm(∇f))
    return k
end
function Newton(x)
     k = x
     ∇f = ForwardDiff.gradient(f,k)
     H = ForwardDiff.hessian(f,k) 
     iter = 0
     while norm(∇f) > 1e-4
        if det(H) == 0
            error("Hessiana é singular")
        end
        d = -H \∇f
        ∇f = ForwardDiff.gradient(f,k)
        H = ForwardDiff.hessian(f,k) 
        k = k + d
        iter +=1
        if iter > 1000
            error("Muita paçoca pouca resposta")
        end
     end
     println(norm(∇f))
     println(iter)
     return k
end


Y = Cauchy(P_i)
println("O ponto minizador local encontrado é " , Y)
A = ForwardDiff.hessian(f,Y)
println("A Hessiana do ponto é ", A)
println("De Autovalores e autovetores: ",eigen(A))