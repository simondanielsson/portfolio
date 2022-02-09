/*
 * Main file executed when running ./gradlew. 
 */
package project1;

import project1.Hardware.Computer.*;
import project1.Hardware.Memory.*;
import project1.Hardware.Operand.Word.WordFactory.WordFactory;
import project1.Hardware.Program.*;
import project1.Software.Program.*;
import project1.Software.WordFactory.*;

public class App {

    public static void main(String[] args) {
        WordFactory lwf = new LongWordFactory();
        WordFactory bwf = new ByteWordFactory();
        run("factorial(5) med LongWord", new Factorial("5", lwf), lwf);
        run("factorial(5) med ByteWord", new Factorial("5", bwf), bwf);
        run("factorial(16) med LongWord", new Factorial("16", lwf), lwf);
        run("factorial(16) med ByteWord", new Factorial("16", bwf), bwf);
        run("sum(5) med LongWord", new Sum("5", lwf), lwf);
        run("sum(5) med ByteWord", new Sum("5", bwf), bwf);
        run("sum(16) med LongWord", new Sum("16", lwf), lwf);
        run("sum(16) med ByteWord", new Sum("16", bwf), bwf);
    }

    static void run(String message, Program program, WordFactory wf) {
        System.out.println("================================");
        System.out.println(message);
        System.out.println(program);
        Computer computer = new Computer(new Memory(1024, wf));
        computer.load(program);
        computer.run();
    }
}
