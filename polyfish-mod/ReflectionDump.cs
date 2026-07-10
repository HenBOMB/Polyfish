using System;
using System.Reflection;
using System.Linq;

class Program
{
    static void Main(string[] args)
    {
        try
        {
            var assemblyPath = "/mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-mod/decompiled/PolytopiaBackendBase.dll";
            var assembly = Assembly.LoadFrom(assemblyPath);
            
            foreach (var type in assembly.GetTypes().Where(t => t.Name.Contains("Command")))
            {
                Console.WriteLine($"Type: {type.FullName}");
                foreach (var ctor in type.GetConstructors())
                {
                    var paramStrs = ctor.GetParameters().Select(p => $"{p.ParameterType.Name} {p.Name}");
                    Console.WriteLine($"  Constructor: {type.Name}({string.Join(", ", paramStrs)})");
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine(ex.ToString());
        }
    }
}
