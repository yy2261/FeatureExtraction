package visitor;

import java.io.BufferedWriter;

import org.eclipse.jdt.core.dom.ASTVisitor;
import org.eclipse.jdt.core.dom.MethodInvocation;
import org.eclipse.jdt.core.dom.ClassInstanceCreation;
import org.eclipse.jdt.core.dom.MethodDeclaration;
import org.eclipse.jdt.core.dom.TypeDeclaration;
import org.eclipse.jdt.core.dom.EnumDeclaration;
import org.eclipse.jdt.core.dom.WhileStatement;
import org.eclipse.jdt.core.dom.CatchClause;
import org.eclipse.jdt.core.dom.ThrowStatement;
import org.eclipse.jdt.core.dom.IfStatement;
import org.eclipse.jdt.core.dom.ForStatement;
import org.eclipse.jdt.core.dom.SwitchStatement;
import org.eclipse.jdt.core.dom.TryStatement;
import org.eclipse.jdt.core.dom.BreakStatement;
import org.eclipse.jdt.core.dom.ContinueStatement;

  
public class DemoVisitor extends ASTVisitor {
	
	public BufferedWriter bw;
	
	public void addBw(BufferedWriter bw){
		this.bw = bw;
	}
  
    @Override
    public boolean visit(MethodDeclaration node){
    	try{
    		this.bw.write("Method Declaration: \t" + node.getName()+"\n");  
    	}catch(Exception e){}
    	System.out.println("Done!");
        return true;  
    }  
  
    @Override  
    public boolean visit(TypeDeclaration node) {  
    	try{
    		this.bw.write("Type Declaration: \t" + node.getName()+"\n");  
    	}catch(Exception e){}
    	System.out.println("Done!");
        return true;  
    }
    
    @Override  
    public boolean visit(EnumDeclaration node) {
    	try{
    		this.bw.write("Enum Declaration: \t" + node.getName()+"\n");  
    	}catch(Exception e){}
    	System.out.println("Done!");
        return true;  
    }  
    
    @Override
    public boolean visit(MethodInvocation node) {
    	try{
    		this.bw.write("Method Invocation: \t" + node.getName()+"\n");  
    	}catch(Exception e){}
    	System.out.println("Done!");
    	return true;
    }
    
    @Override
    public boolean visit(ClassInstanceCreation node) {
    	try{
    		this.bw.write("Class Ins Creation: \t" + node.getType()+"\n");  
    	}catch(Exception e){}
    	System.out.println("Done!");
    	return true;
    }
    
    @Override
    public boolean visit(WhileStatement node) {
    	try{
    		this.bw.write("While Statement\n");  
    	}catch(Exception e){}
    	System.out.println("Done!");
    	return true;
    }
    
    @Override
    public boolean visit(CatchClause node) {
    	try{
    		this.bw.write("Catch Clause\n");  
    	}catch(Exception e){}
    	System.out.println("Done!");
    	return true;
    }
    
    @Override
    public boolean visit(ThrowStatement node) {
    	try{
    		this.bw.write("Throw Statement\n");  
    	}catch(Exception e){}
    	System.out.println("Done!");
    	return true;
    }
    
    @Override
    public boolean visit(IfStatement node) {
    	try{
    		this.bw.write("If Statement\n");  
    	}catch(Exception e){}
    	System.out.println("Done!");
    	return true;
    }
    
    @Override
    public boolean visit(ForStatement node) {
    	try{
    		this.bw.write("For Statement\n");  
    	}catch(Exception e){}
    	System.out.println("Done!");
    	return true;
    }
    
    @Override
    public boolean visit(SwitchStatement node) {
    	try{
    		this.bw.write("Switch Statement\n");  
    	}catch(Exception e){}
    	System.out.println("Done!");
    	return true;
    }
    
    @Override
    public boolean visit(TryStatement node) {
    	try{
    		this.bw.write("Try Statement\n");  
    	}catch(Exception e){}
    	System.out.println("Done!");
    	return true;
    }
    
    @Override
    public boolean visit(BreakStatement node) {
    	try{
    		this.bw.write("Break Statement\n");  
    	}catch(Exception e){}
    	System.out.println("Done!");
    	return true;
    }
    
    @Override
    public boolean visit(ContinueStatement node) {
    	try{
    		this.bw.write("Continue Statement\n");  
    	}catch(Exception e){}
    	System.out.println("Done!");
    	return true;
    }
}  