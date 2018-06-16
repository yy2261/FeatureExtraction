package DemoVisitorTest;

import java.io.File;
import java.io.FileWriter;
import java.io.BufferedWriter;

import org.eclipse.jdt.core.dom.CompilationUnit;  
import util.JdtAstUtil;  
import visitor.DemoVisitor;  
  
public class DemoVisitorTest {
    public static void parse(String path, String printPath) throws Exception{
        File file = new File(printPath);
        if(!file.exists()){
        	file.createNewFile();
        }
        FileWriter fw = new FileWriter(file.getAbsoluteFile(), true);
        BufferedWriter bw = new BufferedWriter(fw);
        CompilationUnit comp = JdtAstUtil.getCompilationUnit(path);
        DemoVisitor visitor = new DemoVisitor();
        visitor.addBw(bw);
        comp.accept(visitor);
        bw.close();
    }
}